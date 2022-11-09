# Dark_DCpose
## Introduction

This is the official code of **Human Action Recognition Algorithm in Dark Environment Based on DCPose.** It is based on Deep Dual Consecutive Network for Human Pose Estimation （CVPR2021）

Human action recognition algorithm is one of the hot topics in the field of computer vision. Although state-of-the-art human action recognition algorithms have performed well in bright environments, their accuracy will be greatly affected when dealing with blurred human boundaries and backgrounds in dark video environments.
In order to extract the incidental features of the video context in the dark environment, a human action recognition network in the dark based on DCPose (Deep Dual Consecutive Network for Human Pose Estimation) is designed and implemented. The network first analyzes the context information in the video sequence through DCPose, extracts the human body key point information, and then uses the multi-layer perceptron to classify the sequence key point information for action classification.
The human pose estimation mAP of the network on the PoseTrack2018 dataset reaches 79.0, and the mAP for action recognition on the ARID test set reaches 51%.


<p align='center'>
	<img src="docs\images\network structure.png" style="zoom:100%;" />
</p>

## Experiment Results

<p align='center'>
	<img src="docs/images/Mean Average Precision by Epochs.png" style="zoom:100%;" />
</p>

## Installation & Quick Start

### Environment
The code is developed using python 3.6.12, pytorch-1.10.2+cu113, and CUDA 11.7 on Windows 10. For my experiments, I used 1 NVIDIA 1070 GPU.

### Installation
1.Create a conda virtual environment and activate it
```
conda create -n DCPose python=3.6.12
conda activate DCPose
```
2.Install dependencies through DCPose_requirements.txt
```
pip install -r DCPose_requirement.txt
```
3.Install DCN
```
cd thirdparty/deform_conv
python setup.py develop
```
4.Download DCPose's [pretrained models and supplementary](https://drive.google.com/drive/folders/1WE76QSeBOimUBT85i387qBujnb0fA5lw?usp=share_link). Put it in the directory ${DCPose_SUPP_DIR}. Also download [MLP's pretrained model](https://drive.google.com/file/d/1rTj5S08MjJaFKQJRZXRzGMldXE3wBg4q/view?usp=share_link).

### Predict on video
Put input video directly in .demo\input.
Your extracted PoseTrack18 images directory should look like this:
```
${Dark_DCPose_DIR}
    |--demo
        |--input
            |-- test
            `-- train
                |--{type 1}
                `--{type 2}
                ......
            `-- val
            `-- predict
```
Predict input video in predict folder.
```
cd demo
python predict.py
```

### Training
I have not completed the transfer training for bone dry HRNet and DCPose.

Training the MLP part:

Put training dataset and validation dataset in train and val folder mentioned above.

Put the label txt in demo folder.
```
cd demo
python video.py
```

# Citation
```
@InProceedings{Liu_2021_CVPR,
    author    = {Liu, Zhenguang and Chen, Haoming and Feng, Runyang and Wu, Shuang and Ji, Shouling and Yang, Bailin and Wang, Xun},
    title     = {Deep Dual Consecutive Network for Human Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {525-534}
}
```
