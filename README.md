# Challenging Fake Image Detection using GAN Models (Problem - 1)
## Problem Statement
Detecting fake or manipulated images in today's digital age has become increasingly
challenging due to the advancements in Generative Adversarial Networks (GANs). These
AI-powered tools have made it easier than ever to create convincing fake images that can
deceive both human observers and traditional image analysis techniques. The problem at hand
is to develop a robust and effective fake image detection system that can differentiate between
genuine and manipulated images generated by GAN models.

## Dataset
The dataset was downloaded from Kaggle. Click [here](https://www.kaggle.com/datasets/uditsharma72/real-vs-fake-faces) for the reference.
It is divided into two subfolders, fake and real representing the fake images and real images respectively.  
The dataset consists of 1081 real images and 960 fake images

## Setup
PLease download the `detector_model.h5` file to access the saved model  
Download the `detector.py` file.   
On `Line 8` replace the path of saved model with the actual path where the downloaded file was saved on your system (Most probably in your Downloads folder)

## Running the detector
On your terminal, run the [detector.py](https://github.com/fuzail1011/fake_detect_system/blob/nothemain/detector.py) file as `python detector.py "your_test_image_location.jpg"`  
Replace your_test_image_location with the actual location of the image that you want to detect as real or fake  
Note: The image format has to be in `.jpg`

## Model
The model used is a `ResNet-50`  
It is an improved CNN which is 50 layers deep. This model is excellent for image processing, and can evaluate images with a high accuracy.  
The model is implemented using Tensorflow and Keras libraries. It was run for 40 epochs, after which the accuracy of the model was consistently above 0.95

Model Architecture  
- Input Shape: (224, 224, 3) - Typically, RGB images of size 224x224 pixels.
- Total Number of Parameters: 25,636,712
- Total Number of Trainable Parameters: 25,583,592
- Total Number of Non-trainable Parameters: 53,120

The model was trained on an Nvidia GeForce GTX 1650 Ti, with CUDA version 11.7  

## Additional Model Insights
Attempt - 1 : Using a CNN with 15 layers  
The accuracy obtained from this model was around 0.5 and hence not sufficient to give a good result. This model was rejected.  
Attempt - 2 : Using a Discriminator GAN model  
The accuracy obtained in this model was around 0.4, which is not sufficient and hence rejected.  
Attempt - 3 : Using the ResNet-50, which gives the best accuracy and therfore was finalized for model creation to solve the problem.  

## References  
- https://www.ijcrt.org/papers/IJCRT2005044.pdf
- https://philarchive.org/archive/SALCOR-3
- E. R. S. de Rezende, G. C. S. Ruppert and T. Carvalho, "Detecting Computer Generated Images with Deep Convolutional Neural Networks," 2017 30th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), Niteroi, Brazil, 2017, pp. 71-78, doi: 10.1109/SIBGRAPI.2017.16.  URL - (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8097296&isnumber=8097279)
- [ResNet-50: The Basics and a Quick Tutorial - Datagen](https://datagen.tech/guides/computer-vision/resnet-50/#:~:text=ResNet%2D50%20Architecture,-The%20original%20ResNet&text=It%20provided%20a%20novel%20way,network%20to%20a%20residual%20network.)
