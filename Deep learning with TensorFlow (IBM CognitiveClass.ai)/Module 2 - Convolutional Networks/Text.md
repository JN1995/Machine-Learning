# Convolutional Networks

## Learning objectives

**In this lesson we will learn about:**

* Introduction to Convolutional Networks
* Convolution and Feature Learning 
* Convoltion with Python and Tensorflow
* The MNIST Database
* Multilayer Perceptron with Tensorflow
* Convolutional Network with Tensorflow

## Introduction to Convolutional Networks


**The original goal of the CNN was to form the best possible representation of our visual world, in order to support recognition tasks.**

The CNN solution needed to have two key features to make it viable:

* it needed to be able to detect the objects in the image and place them into the appropriate category.
* it also needed to be robust against differences in pose, scale, illumination, conformation, and clutter.

**interesting enough, the solution to the object recognition issue was inspired by examining the way our own visual cortex operates.**

CNN starts with an input image, it then extracts a few primitive features, and combines those features to form certain parts of the object and finally, it pulls together all of the various parts to form the object itself. 

**In essence, it is a hierarchical way of seeing objects.** That is, in the first layer, very simple features are detected. Then, these are combined to shape more complicated features in the second layer, and so on to detect objects.

the CNN, is a set of layers, with each of them being responsible to detect a set of feature sets. And these features are going to be more abstract as it goes further into examining the next layer.












