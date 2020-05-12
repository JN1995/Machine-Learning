# Recurrent Neural Networks

## Learning objectives

**In this lesson we will learn about:**

* The Recurrent Neural Network Model 
* Long Short-Term Memory
* Recursive Neural Tensor Network Theory
* Applying Recurrent Networks to Language Model


## The Sequential Problem

Whenever the points in a dataset are dependent on the other points, the data is said to be sequential.

A common example of this is a time series, such as a stock price or sensor data, where each data point represents an observation at a certain point in time. 

A Recurrent Neural Network has a mechanism that can handle a sequential dataset.


## The RNN model

A Recurrent neural network, or RNN for short, is a great tool for modeling sequential data.

The RNN is able to remember the analysis that was done up to a given point by maintaining a state, or a context, so to speak. You can think of the “state” as the “memory” of RNN, which captures information about what’s been previously calculated.

This “state” recurs back into the net with each new input, which is where the network gets its name

**how this works**

in the hidden layer, two values will be calculated:

* First, the new or updated state, denoted as h_new, is to be used for the next data point in the sequence. 
* And second, the output of the network will be computed, which is denoted as y. 

**Recurrent neural networks are extremely versatile and are used in a wide range of applications that deal with sequential data.**

* One of these applications is speech recognition. As you can see, it is a type of a many-to-many network. That is, the goal is to consume a sequence of data and then produce another sequence.

* Another application of RNN is image captioning. Although it’s not purely recurrent, you can create a model that’s capable of understanding the elements in an image. Then, using the RNN, you can string the elements as words together to form a caption that describes the scene. Typically, RNN has outputs at each time step, but it depends on the problem that RNN is addressing. There is one input as image, and the output is a sequence of words. So, it is sometimes called one-to-many.

* RNN can also be Many-to-one, that is, it consumes a sequence of data and produces just one output.

**Despite all of its strengths, the recurrent neural network is not a perfect model.** (Recurrent Neural Network Problems)

* One issue is that the network needs to keep track of the states at any given time. There could be many units of data, or many time steps, so this becomes computationally expensive. One compromise is to only store a portion of the recent states in a time window. 

* Another issue is that Recurrent Neural Networks are extremely sensitive to changes in their parameters. (gradient descent optimizers may struggle to train the net; the net may suffer from the “Vanishing Gradient” problem, where the gradient drops to nearly zero and training slows to a halt;  it may also suffer from the “Exploding Gradient”, where the gradient grows exponentially off to infinity.)















