# Unet Semantic Segmentation for Cracks

## Real time Crack Segmentation using PyTorch, OpenCV, ONNX runtime

### Dependencies

<p> Pytorch <br>
<p> OpenCV <br>
<p> ONNX runtime <br>
<p> CUDA >= 9.0 <br>

### Instructions

<p> 1.Train model with your datatset and save model weights (.pt file) using unet_train.py on supervisely.ly <br>
<p> 2.Convert model weights to ONNX format using pytorch_to_onnx.py <br>
<p> 3.Obtain real time inference using crack_det_new.py <br>

Crack segmentation model files can be downloaded by clicking this [link](https://drive.google.com/file/d/10dSDs6riOSb4dWPtEDRCoqyOtO_Uh7k8/view?usp=sharing)


### Results:

![alt text](https://raw.githubusercontent.com/anishreddy3/Crack_Semantic_Segmentation/master/crack_inference.gif)


### Graphs:
![alt text](https://raw.githubusercontent.com/anishreddy3/Crack_Semantic_Segmentation/master/accuracy.png)

![alt text](https://raw.githubusercontent.com/anishreddy3/Crack_Semantic_Segmentation/master/loss.png)



