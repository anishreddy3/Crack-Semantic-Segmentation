"""
Usage: Used to inference on images available in a folder

python crack_det_pt.py -c "class file" -l "color file"  -idir "dataset directory" -odir "output directory" -m model file

python crack_det_pt.py -c unet_classes.txt -l unet_colors.txt -idir "Dataset/sample dataset/" -odir "output_test/" -m model_files/model.pt
"""

from collections import OrderedDict
from pytorch_to_onnx import construct_unet
import numpy as np
from PIL import Image, ImageDraw
import cv2
import imutils
import argparse
import datetime
import time
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
import os



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



def predict_image(img, image1, outdir,filename, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    st = time.time()
    yb = model(xb)
    end = time.time()
    elap = end-st
    print("[INFO] single frame took {:.4f} seconds".format(elap))

    pred = yb.cpu().detach().numpy()
    print(pred.shape)

    # Infer the total number of classes, height and width
    (numClasses, height, width) = pred.shape[1:4]

    print(numClasses, height, width)

    # Argmax is utilized to find the class label with largest probability for every pixel in the image
    classMap = np.argmax(pred[0], axis=0)

    # classes are mapped to their respective colours
    mask = COLORS[classMap]

    frame = image1
    # resizing the mask and class map to match its dimensions with the input image
    mask = cv2.resize(
        mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    classMap = cv2.resize(
        classMap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Construct weighted combination of input image along with mask to form output visualization
    output = ((0.4 * frame) + (0.6 * mask)).astype("uint8")
    print(outdir+filename)
    cv2.imwrite(outdir+filename, output)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--classes", required=True,
                    help="path to .txt file containing class labels")
    ap.add_argument("-l", "--colors", type=str,
                    help="path to .txt file containing colors for labels")
    ap.add_argument("-idir", "--input_directory", required=True,
                    help="path to input directory")
    ap.add_argument("-odir", "--output_directory", required=True,
                    help="path to output directory")
    ap.add_argument("-m", "--model_file", required=True,
                    help="path to model file")

    args = vars(ap.parse_args())

    # load the class label names
    CLASSES = open(args["classes"]).read().strip().split("\n")

    # if a colors file was supplied, load it from disk
    if args["colors"]:
        COLORS = open(args["colors"]).read().strip().split("\n")
        COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
        COLORS = np.array(COLORS, dtype="uint8")

    device = get_default_device()
    unet = construct_unet(2)

    # num_classes=2 , heating and nonheating
    model_unet = to_device(unet, device)
    model_file = args["model_file"]
    state_dict = torch.load(model_file)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    # load params
    model_unet.load_state_dict(new_state_dict)
    model_unet.eval()

    directory = args["input_directory"]

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image = Image.open(directory+filename)
            image1 = cv2.imread(directory+filename)
            test_transformations = transforms.Compose([
                transforms.Resize((256, 256)),  # resize input images to 255,255
                transforms.ToTensor()
            ])

            transformed_img = test_transformations(image)

            pred_result = predict_image(
                transformed_img, image1, args["output_directory"], filename, model_unet)
    print("Completed")
