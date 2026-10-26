# Crack Semantic Segmentation

Real-time crack segmentation with **PyTorch**, **OpenCV**, and **ONNX Runtime**.

Train a U-Net, export to ONNX, then run folder or video inference.

![Crack inference demo](crack_inference.gif)

## Project layout

| Path | Role |
| --- | --- |
| `unet_train.py` | Train U-Net and save `.pt` weights |
| `pytorch_to_onnx.py` | Export a checkpoint to ONNX |
| `crack_inference_folder.py` | Batch image inference |
| `crack_det_new.py` | Video inference with an ONNX model |
| `sample_dataset/` | Sample images for a quick smoke run |
| `unet_classes.txt` / `unet_colors.txt` | Class and color maps |

## Dependencies

Install a current PyTorch build for your platform, then:

```bash
pip install -r requirements.txt
```

The original 2019 environment used Python 3.6 + CUDA 10 wheels. Those pins remain commented in `requirements.txt` for archival reproduction only.

## Workflow

1. Train on your dataset and save weights (`.pt`) with `unet_train.py` (originally used with Supervise.ly).
2. Convert weights to ONNX:

   ```bash
   python pytorch_to_onnx.py --weights path/to/model.pt --output model_files/model.onnx
   ```

3. Run folder or video inference.

Pretrained model files: [Google Drive](https://drive.google.com/file/d/10dSDs6riOSb4dWPtEDRCoqyOtO_Uh7k8/view?usp=sharing)

## Commands

Folder inference:

```bash
python crack_inference_folder.py \
  -c unet_classes.txt \
  -l unet_colors.txt \
  -idir "sample_dataset/" \
  -odir "output_test/" \
  -m model_files/model.pt
```

Video inference:

```bash
python crack_det_new.py \
  -c unet_classes.txt \
  -l unet_colors.txt \
  -i input_video.mp4 \
  -o output_video.mp4 \
  -m model_files/model.onnx
```

## Training curves

![Accuracy](accuracy.png)

![Loss](loss.png)

## Notes

- Paths are CLI arguments only — no machine-local absolute paths are required.
- Large model weights are not committed; download them from the Drive link above.
- This repository is a portfolio snapshot of a working 2019 pipeline, not an actively maintained product.

## License

MIT — see [LICENSE](LICENSE).
