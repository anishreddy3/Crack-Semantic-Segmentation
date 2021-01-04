# Unet Semantic Segmentation for Cracks

## Real time Crack Segmentation using PyTorch, OpenCV, ONNX runtime

### Dependencies:

<p> Pytorch <br>
<p> OpenCV <br>
<p> ONNX runtime <br>
<p> CUDA >= 9.0 <br>

### Instructions:

<p> 1.Train model with your datatset and save model weights (.pt file) using unet_train.py on supervisely.ly <br>
<p> 2.Convert model weights to ONNX format using pytorch_to_onnx.py <br>
<p> 3.Obtain real time inference using crack_det_new.py <br>

Crack segmentation model files can be downloaded by clicking this [link](https://drive.google.com/file/d/10dSDs6riOSb4dWPtEDRCoqyOtO_Uh7k8/view?usp=sharing)

### Commands

Usage: Used to inference on images available in a folder on GPU

    python crack_inference_folder.py -c "class file" -l "color file"  -idir "dataset directory" -odir "output directory" -m model file

    python crack_inference_folder.py -c unet_classes.txt -l unet_colors.txt -idir "Dataset/sample dataset/" -odir "output_test/" -m model_files/model.pt

Usage: Used to inference on images available in a folder on CPU

    python crack_det_new.py -c "class file" -l "color file"  -i "input video" -o "output video" -m model file

    python crack_det_new.py -c unet_classes.txt -l unet_colors.txt -i "input_vdo.mp4 -odir "output_vdo.mp4" 

### Results:

![](crack_inference.gif)


### Graphs:
![alt text](https://raw.githubusercontent.com/anishreddy3/Crack_Semantic_Segmentation/master/accuracy.png)

![alt text](https://raw.githubusercontent.com/anishreddy3/Crack_Semantic_Segmentation/master/loss.png)



