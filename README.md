# CycleGan_Image_translation_RGB_IR
Images are the most data types used in computer vision. For som topics like face recognition, despite RGB images, other types of images can be used such depth images and infrared images. Nowadays, networks are built to take as input multimodalities,. we mostly see RGB and Depth images combination. Infrared images are less udes due to its collection difficulties. In this project, WE used an adversarial network, cyclegan to generate infreared images forom RGB images.

# Dataset.
The databased used for training is the Lock3DFace databased. It is a arge-scale database consisting of low cost Kinect 3D face videos. Mostly used to solve face topics in computer vision. The dataset contains 519 subjects faces in RGB, deppth and infrared format.
![lock3d_images](https://user-images.githubusercontent.com/48753146/150793755-72da2ffd-1f12-43a7-aaf5-dd6bd3ec9f50.PNG)

# Network
Cyclegan has been  used to translate RGB to IR. The goal of cyclegan is to learn mapping between an input image and an output image using a training set of aligned image pairs. Different from pix2pix network that requires paired images, cyclegan can work with unapired images. The network is composed of two generator models: one generator (Generator-A) for generating images for the first domain (Domain-A) and the second generator (Generator-B) for generating images for the second domain (Domain-B).

![cyclegan](https://user-images.githubusercontent.com/48753146/150794789-2bdd8ef8-7bbe-40b3-ad9c-93cba71b7a78.png)

# Results

## training result, RGB to infrared
![AtoB_generated_plot_011480](https://user-images.githubusercontent.com/48753146/150796370-6169d282-7886-445f-a03c-66102cf7c23d.png)

## testing result
![Capture_github](https://user-images.githubusercontent.com/48753146/150797842-5ea65d10-8daa-403d-a58f-1e60e9d8a651.PNG)

# How to train
1. Download the Lock3D face dataset or any dataset for the translation. split each domain into two parts: train and test.
2. Preprocess the dataset to generate an npz file.
3. Train by using the train.py file
4. Adjust the parameters according to your computer ressources.
5. Test your model



References
Dataset original paper: http://irip.buaa.edu.cn/lock3dface/resources/Lock3DFace.pdf

network implementation: https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
