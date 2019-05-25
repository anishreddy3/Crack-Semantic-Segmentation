# Traing Unet Model on Supervisely

import os
from collections import defaultdict

import cv2
import torch
import torch.nn.functional as functional
from torch.optim import Adam
from torch.utils.data import DataLoader

from supervisely_lib import logger
from supervisely_lib.nn.pytorch.metrics import MultiClassAccuracy
from supervisely_lib.nn.pytorch.weights import WeightsRW

import supervisely_lib as sly
from supervisely_lib.nn.hosted.class_indexing import CONTINUE_TRAINING, TRANSFER_LEARNING
from supervisely_lib.nn.hosted.trainer import SuperviselyModelTrainer
from supervisely_lib.nn.pytorch.cuda import cuda_variable
from supervisely_lib.nn.training.eval_planner import EvalPlanner
from supervisely_lib.nn.dataset import ensure_samples_nonempty
from supervisely_lib.task.progress import epoch_float

import config as config_lib
from common import create_model, UnetJsonConfigValidator
from dataset import UnetV2Dataset
from metrics import Dice, NLLLoss, BCEDiceLoss
from debug_saver import DebugSaver


# decrease lr after 'patience' calls w/out loss improvement
#TODO@ Factor out policy from here and smarttool.
class LRPolicyWithPatience:
    def __init__(self, optim_cls, init_lr, patience, lr_divisor, model):
        self.optimizer = None
        self.optim_cls = optim_cls
        self.lr = init_lr
        self.patience = patience
        self.lr_divisor = lr_divisor
        self.losses = []
        self.last_reset_idx = 0

        sly.logger.info('Selected optimizer.', extra={'optim_class': self.optim_cls.__name__})
        self._reset(model)

    def _reset(self, model):
        self.optimizer = self.optim_cls(model.parameters(), lr=self.lr)
        sly.logger.info('Learning Rate has been updated.', extra={'lr': self.lr})

    def reset_if_needed(self, new_loss, model):
        self.losses.append(new_loss)
        no_recent_update = (len(self.losses) - self.last_reset_idx) > self.patience
        no_loss_improvement = min(self.losses[-self.patience:]) > min(self.losses)
        if no_recent_update and no_loss_improvement:
            self.lr /= float(self.lr_divisor)
            self._reset(model)
            self.last_reset_idx = len(self.losses)


class UnetV2Trainer(SuperviselyModelTrainer):
    @staticmethod
    def get_default_config():
        return {
            'dataset_tags': {
                'train': 'train',
                'val': 'val',
            },
            'batch_size': {
                'train': 6,
                'val': 3,
            },
            'data_workers': {
                'train': 0,
                'val': 3,
            },
            'allow_corrupted_samples': {
                'train': 0,
                'val': 0,
            },
            'special_classes': {
                'background': 'bg',
                'neutral': 'neutral',
            },
            'input_size': {
                'width': 256,
                'height': 256,
            },
            'epochs': 3,
            'val_every': 0.5,
            'lr': 0.1,
            'momentum': 0.9,
            'lr_decreasing': {
                'patience': 1000,
                'lr_divisor': 5,
            },
            'loss_weights': {
                'bce': 1.0,
                'dice': 1.0,
            },
            'weights_init_type': TRANSFER_LEARNING,  # CONTINUE_TRAINING,
            'validate_with_model_eval': True,
            'gpu_devices': [0],
        }

    def __init__(self):
        self.bkg_input_idx = 0
        super().__init__(default_config=UnetV2Trainer.get_default_config())

    @property
    def class_title_to_idx_key(self):
        return config_lib.class_to_idx_config_key()

    @property
    def train_classes_key(self):
        return config_lib.train_classes_key()

    def _validate_train_cfg(self, config):
        UnetJsonConfigValidator().validate_train_cfg(config)

    def _determine_config(self):
        super()._determine_config()
        self.device_ids = self.config['gpu_devices']
        sly.env.remap_gpu_devices(self.device_ids)

    def _determine_model_classes(self):
        super()._determine_model_classes_segmentation(bkg_input_idx=self.bkg_input_idx)
        self.class_title_to_idx_with_internal_classes = self.class_title_to_idx.copy()
        self.neutral_idx = None
        neutral_title = self.config['special_classes'].get('neutral', None)
        if neutral_title is not None:
            self.neutral_idx = max(self.class_title_to_idx_with_internal_classes.values()) + 1
            self.class_title_to_idx_with_internal_classes[neutral_title] = self.neutral_idx

    def _construct_and_fill_model(self):
        # TODO: Move it progress to base class
        progress_dummy = sly.Progress('Building model:', 1)
        progress_dummy.iter_done_report()
        self.model = create_model(n_cls=(max(self.class_title_to_idx.values()) + 1), device_ids=self.device_ids)

        if sly.fs.dir_empty(sly.TaskPaths.MODEL_DIR):
            sly.logger.info('Weights will not be inited.')
            # @TODO: add random init (m.weight.data.normal_(0, math.sqrt(2. / n))
        else:
            wi_type = self.config['weights_init_type']
            ewit = {'weights_init_type': wi_type}
            sly.logger.info('Weights will be inited from given model.', extra=ewit)

            weights_rw = WeightsRW(sly.TaskPaths.MODEL_DIR)
            if wi_type == TRANSFER_LEARNING:
                self.model = weights_rw.load_for_transfer_learning(self.model, ignore_matching_layers=['last_conv'],
                                                                   logger=logger)
            elif wi_type == CONTINUE_TRAINING:
                self.model = weights_rw.load_strictly(self.model)

            sly.logger.info('Weights are loaded.', extra=ewit)

    def _construct_loss(self):
        self.metrics = {
            'accuracy': MultiClassAccuracy(ignore_index=self.neutral_idx)
        }

        if len(self.out_classes) == 2:
            sly.logger.info('Binary segmentation, will use both BCE & Dice loss components.')
            self.metrics.update({
                'dice': Dice(ignore_index=self.neutral_idx)
            })
            l_weights = self.config['loss_weights']
            self.criterion = BCEDiceLoss(ignore_index=self.neutral_idx, w_bce=l_weights['bce'],
                                         w_dice=l_weights['dice'])
        else:
            sly.logger.info('Multiclass segmentation, will use NLLLoss only.')
            self.criterion = NLLLoss(ignore_index=self.neutral_idx)

        self.val_metrics = {
            'loss': self.criterion,
            **self.metrics
        }
        sly.logger.info('Selected metrics.', extra={'metrics': list(self.metrics.keys())})

    def _construct_data_loaders(self):
        src_size = self.config['input_size']
        input_size = (src_size['height'], src_size['width'])

        self.pytorch_datasets = {}
        for the_name, the_tag in self.name_to_tag.items():
            samples_lst = self._deprecated_samples_by_tag[the_tag]
            ensure_samples_nonempty(samples_lst, the_tag, self.project.meta)

            the_ds = UnetV2Dataset(
                project_meta=self.project.meta,
                samples=samples_lst,
                out_size=input_size,
                class_mapping=self.class_title_to_idx_with_internal_classes,
                bkg_color=self.bkg_input_idx,
                allow_corrupted_cnt=self.config['allow_corrupted_samples'][the_name]
            )

            self.pytorch_datasets[the_name] = the_ds

            sly.logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

        self.data_loaders = {}
        for name, need_shuffle in [
            ('train', True),
            ('val', False),
        ]:
            # note that now batch_size from config determines batch for single device
            batch_sz = self.config['batch_size'][name]
            batch_sz_full = batch_sz * len(self.device_ids)
            n_workers = self.config['data_workers'][name]
            self.data_loaders[name] = DataLoader(
                dataset=self.pytorch_datasets[name],
                batch_size=batch_sz_full,  # it looks like multi-gpu validation works
                num_workers=n_workers,
                shuffle=need_shuffle,
            )
        sly.logger.info('DataLoaders are constructed.')

        self.train_iters = len(self.data_loaders['train'])
        self.val_iters = len(self.data_loaders['val'])
        self.epochs = self.config['epochs']
        self.eval_planner = EvalPlanner(epochs=self.epochs, val_every=self.config['val_every'])

    def _dump_model_weights(self, out_dir):
        WeightsRW(out_dir).save(self.model)

    def _validation(self):
        sly.logger.info("Before validation", extra={'epoch': self.epoch_flt})
        if self.config['validate_with_model_eval']:
            self.model.eval()

        metrics_values = defaultdict(int)
        samples_cnt = 0

        for val_it, (inputs, targets) in enumerate(self.data_loaders['val']):
            inputs, targets = cuda_variable(inputs, volatile=True), cuda_variable(targets)
            outputs = self.model(inputs)
            full_batch_size = inputs.size(0)
            for name, metric in self.val_metrics.items():
                metric_value = metric(outputs, targets)
                if isinstance(metric_value, torch.autograd.Variable):  # for val loss
                    metric_value = metric_value.data[0]
                metrics_values[name] += metric_value * full_batch_size
            samples_cnt += full_batch_size

            sly.logger.info("Validation in progress", extra={'epoch': self.epoch_flt,
                                                         'val_iter': val_it, 'val_iters': self.val_iters})

        for name in metrics_values:
            metrics_values[name] /= float(samples_cnt)

        sly.report_metrics_validation(self.epoch_flt, metrics_values)

        self.model.train()
        sly.logger.info("Validation has been finished", extra={'epoch': self.epoch_flt})
        return metrics_values

    def train(self):
        progress = sly.Progress('Model training: ', self.epochs * self.train_iters)
        self.model.train()

        lr_decr = self.config['lr_decreasing']
        policy = LRPolicyWithPatience(
            optim_cls=Adam,
            init_lr=self.config['lr'],
            patience=lr_decr['patience'],
            lr_divisor=lr_decr['lr_divisor'],
            model=self.model
        )
        best_val_loss = float('inf')

        debug_saver = None
        debug_save_prob = float(os.getenv('DEBUG_PATCHES_PROB', 0.0))
        if debug_save_prob > 0:
            target_multi = int(255.0 / len(self.out_classes))
            debug_saver = DebugSaver(odir=os.path.join(sly.TaskPaths.DEBUG_DIR, 'debug_patches'),
                                     prob=debug_save_prob,
                                     target_multi=target_multi)

        for epoch in range(self.epochs):
            sly.logger.info("Before new epoch", extra={'epoch': self.epoch_flt})

            for train_it, (inputs_cpu, targets_cpu) in enumerate(self.data_loaders['train']):
                inputs, targets = cuda_variable(inputs_cpu), cuda_variable(targets_cpu)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if debug_saver is not None:
                    out_cls = functional.softmax(outputs, dim=1)
                    debug_saver.process(inputs_cpu, targets_cpu, out_cls.data.cpu())

                policy.optimizer.zero_grad()
                loss.backward()
                policy.optimizer.step()

                metric_values_train = {'loss': loss.data[0]}
                for name, metric in self.metrics.items():
                    metric_values_train[name] = metric(outputs, targets)

                progress.iter_done_report()

                self.epoch_flt = epoch_float(epoch, train_it + 1, self.train_iters)
                sly.report_metrics_training(self.epoch_flt, metric_values_train)

                if self.eval_planner.need_validation(self.epoch_flt):
                    metrics_values_val = self._validation()
                    self.eval_planner.validation_performed()

                    val_loss = metrics_values_val['loss']
                    model_is_best = val_loss < best_val_loss
                    if model_is_best:
                        best_val_loss = val_loss
                        sly.logger.info('It\'s been determined that current model is the best one for a while.')

                    self._save_model_snapshot(model_is_best, opt_data={
                        'epoch': self.epoch_flt,
                        'val_metrics': metrics_values_val,
                    })

                    policy.reset_if_needed(val_loss, self.model)

            sly.logger.info("Epoch was finished", extra={'epoch': self.epoch_flt})


def main():
    cv2.setNumThreads(0)  # important for pytorch dataloaders
    x = UnetV2Trainer()  # load model & prepare all
    x.train()


if __name__ == '__main__':
    sly.main_wrapper('UNET_V2_TRAIN', main)
