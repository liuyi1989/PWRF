# DMTNet: DMTNet:A Disentangled Multimodal Transformer Network for V-D-T Salient Object Detection

![image](https://github.com/user-attachments/assets/4dc1dc79-671d-4ac6-8641-fbea76cbe846)


![](./figs/Overview.png)

## Requirements
python 3.9

pytorch 1.11.0

tensorboardX 2.5
## Results and Saliency maps
We provide [saliency maps](https://drive.google.com/file/d/1USRmpamaV5RJyI3iWp3J12bIZ0i-kbjr/view?usp=sharing) of our CATNet on 7 datasets.
## Training
Please run 
```
CatNet_train.py
```
## Pre-trained model and testing
- Download the following pre-trained models ([Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)) and put them in /pre.
- Modify pathes of pre-trained models and datasets.
- Run 
```
CatNet_test.py
```
## Downloading Training and Testing Datasets:
- Download the [training set](https://drive.google.com/file/d/1BPt09rbgSYQcu0LpQoKSNVgXA3aYvkF7/view?usp=sharing) used for training.
- Download the [testing sets](https://drive.google.com/file/d/1wAVNEYDrTZK7oWB4J-3MTXmAo1AsVZCj/view?usp=sharing) used for testing.


