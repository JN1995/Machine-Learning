# Introduction to Tensorflow

## Learning objectives

**In this lesson we will learn about:**

* Introduction to Tensorflow
* Linear, Nonlinear and Logistic Regression with Tensorflow
* Activation Functions

## Introduction to Tensorflow:

**We will be going through a brief overview of the structure and capabilities of the Tensorflow library.**

* Tensorflow is an open source library developed by the Google Brain Team.
* It's an extremely versatile library, originally created for tasks that require heavy numerical computations.
* For this reason, Tensorflow was geared towards the problems of machine learning and deep neural networks.
* Due to a C/C++ backend, Tensorflow is able to run faster than pure Python code.
* Tensorflow application uses a structure known as a data flow graph, which is very useful to first build and then execute it in a session.
* It is also a very common programming model for parallel computing.
* Tensorflow provides both a Python and a C++ API, but Python API is more complete and generally easier to use.
* Tensorflow supports CPUs, GPUs, and even distributed processing in a cluster. It's a very important features as you can train a neural network using CPU and multiple GPUs, which makes the models very efficient on large-scale systems.
* Tensorflow's structure is based on the execution of a data flow graph.


**Data flow graph** has two basic units: 

* The nodes that represent a mathematical operation
* the edges which represent the multi-dimensional arrays, known as a tensor.

==> **The standard usage is to build a graph first and then execute it in a session.**


**What is a Tensor?**

* the data that is passed between the operations a Tensor.
* In effect, a Tensor is a multidimensional array. (such as scalar values is zero dimensional, one dimensional a vector and so on.)
==> **The tensor structure helps us by giving us the freedom to shape the dataset the way we want.**

**In graph:**

* the nodes are called operations, which represent units of computation.
* the edges are tensors which represent the data consumed or produced by an operation.
* feature matrix is a placeholder.

**Placeholder:**

* Placeholders can be seen as "holes" in our model -- through which we can pass the data from outside of the graph.
* Placeholder allow us to create our operations in the graph, without needing the data. When we want to execute the graph, we have to feed placeholders with our input data.
* We need to initialize placeholders before using them.


**Variables:**

* In the graph, Weight Matrix is a variable.
* Tensorflow variables, are used to share and persist some values, that are manipulated by the program.

**Please notice that when we define a placeholder or variable, Tensorflow adds an operation to our graph.**

**NOTE:**

* what makes TensorFlow so popular today, is its architecture. TensorFlow’s flexible architecture allows we to deploy computation on one or more CPUs, or GPUs, or on a desktop, server, or even a mobile device. This means we build our program once, and then you can run it easily on different devices.

**So let’s briefly review the reasons why TensorFlow is well-suited for deep learning applications.**

* First, TensorFlow has built-in support for deep learning and neural networks, so it’s easy to assemble a net, assign parameters, and run the training process.
* Second, it also has a collection of simple, trainable mathematical functions that are useful for neural networks.
* Finally, deep learning as a gradient-based machine learning algorithm will benefit from TensorFlow’s auto-differentiation and optimizers.


## Introduce to Deep Learning

**Why deep learning** (Deep learning is trying to help:)

* the health care industry on tasks such as cancer detection and drug discovery
* in the internet services and mobile phone industries we can see various app which are using Deep Learning for image/video classification and speech recognition (Google Voice, Apple Siri, Microsoft Skype)
* in the media, entertainment and news we can see applications such as video captioning, real-time translation and personalization or recommendation systems.
* in the development of self driving cars, Deep Learing is helping researchers to overcome significant research problems, such as sign and passenger detection or lane tracking.
* in the security field, Deep Learning is used for face recognition and video surveillance.

**The increasing popularity of Deep Learning today comes from three recent advances in our field:**

* First, in the dramatic increases in computer processing capabilities
* Second, in the availability of massive amounts of data for traing computer systems
* Third, in the advances in machine learning algorithms and research


**What is Deep Learning?**

Deep Learning is a series of supervised, semi-supervised, and unsupervised methods that try to solve some machine learning problems using deep neural networks.

**A deep neural network is a neural network which often has more than two layers, and uses specific mathematical modeling in each layer to process data. Generally speaking, these networks are trying to automatically extract feature sets from data, and this is why, they are mostly used in data types where the feature selection process is difficult, such as when analyzing unstrutured datasets, such as image data, video, sound and text.**

## Deep neural network

To better understand Deep Learning, let's first take a look at different deep neural networks and their applications.
Namely: 
* Convolution Neural Networks (CNN)
* Recurrent Neural Networks (RNN)
* Restricted Boltzmann Machines (RBM)
* Deep Belief Networks (DBN)
* Autoencoders

**Convolution Neural Network**

* can automatically find those features and classify the images for us
* approach that learns directly from samples in a way that is much more effective than traditional Neural networks.
CNNs achieve this type of automatic feature selection and classification through multiple specific layers of sophisticated mathematical operations. Through multiple layers, a CNN learns multiple levels of feature sets at different levels of abstraction.

**Recurrent Neural Network**

* is a type of deep learning approach, that tries to solve the problem of modeling sequential data. Whenever the points in a dataset are dependent on the previous points, the data is said to be sequential
* We simply need to feed the network with the sequential data, it then maintains the context of the data and thus, learns the patterns within the data.
* We can also use RNNs for sentiment analysis.
* RNNs can also be used to predict the next word in a sentence
* Text translation is another example of how RNNs can be used. This task is not based on a word-by-word translation and applying grammar rules. Instead, it is a probability model that has been trained on lots of data where the exact same text is translated into another language.
* Speech-to-text is yet another useful and increasingly common application of RNNs. In this case, the recognized voice is not only based on the word sound; RNNs also use the context around that sound to accurately recognize of the words being spoken into the device’s microphone.

**Restricted Boltzman Machine**

* are used to find the patterns in data in an unsupervised manner. 
* are shallow neural nets that learn to reconstruct data by themselves.
* are very important models, because they can automatically extract meaningful features from a given input, without the need to label them.
* RBMs might not be outstanding if you look at them as independent networks, but they are significant as building blocks of other networks, such as Deep Belief Networks.

RBMs are useful for unsupervised tasks such as:
* feature extraction
* dimensionality reduction
* partern recognition
* recommender systems
* handling missing value
* topic modeling

**Deep Belief Network**

* how they are built on top of RBMs.
* A Deep Belief Network is a network that was invented to solve an old problem in traditional artificial neural networks. (The back-propagation problem, that can often cause “local minima” or “vanishing gradients” issues in the learning process.)
* DBN is built to solve this by the stacking of multiple RBMs.

So, **what are the applications of DBNs?** DBNs are generally used for classification -- same as traditional MLPs.
So, one of the most important applications of DBNs is image recognition.
The important part to remember, here, is that a DBN is a very accurate discriminative classifier.
As such, we don’t need a big set of labeled data to train a Deep Belief Network; in fact, a small set works fine because feature extraction is unsupervised by a stack of RBMs.

**Autoencoder**

* Autoencoders were invented to address the issue of extracting desirable features. 
* much like RBMs, Autoencoders try to recreate a given input, but do so with a slightly different network architecture and learning method.
* Autoencoders take a set of unlabeled inputs, encodes them into short codes, and then uses those to reconstruct the original image, while extracting the most valuable information from the data.
* Autoencoders are employed in some of the largest deep learning applications, especially for unsupervised tasks.

As the encoder part of the network, Autoencoders compress data from the input layer into a
short code -- a method that can be used for “dimensionality reduction” tasks.
Also, in stacking multiple Autoencoder layers, the network learns multiple levels of representation
at different levels of abstraction.

For example, to detect a face in an image, the network encodes the primitive features, like the edges of a face.
Then, the first layer's output goes to the second Autoencoder, to encode the less local features, like the nose, and so on. 
Therefore, it can be used for Feature Extraction and image recognition.








