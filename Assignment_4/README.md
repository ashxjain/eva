## Architectural Basics

Building a Deep Neural Network is not easy. But following certain methodology does make it easy enough to understand it and play with it.

#### Analyze Input Dataset and Normalize it
* Network should be designed keeping dataset in mind. We must look at the dataset and then decide our Architecture
* It is better to perform Augmentation on dataset to reduce Overfitting of the network. But what sort of Augmentation should be selected, is only known once we have visualized the dataset
* Images must be normalized so that network trains faster and quickly comes to a minima

#### Getting Network Architecture Right 
* First don't aim at reducing the number of parameters. Aim at getting the network architecture right. By selecting:
  * Number of Layers - Based on the dataset, we must select the number of layers. If dataset demands learning high number of features, then we must have high number of layers
  * Number of Kernels - Same as above, based on the dataset we decide the number of kernels. More the features to learn, more the kernels
  * 3x3 Convolutions - It is always better to use 3x3 convolutions as it is proven to achieve good accuracy. And also it makes run network faster by reducing the number of parameters as compared to using kernels of large size
  * When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
  * MaxPooling - Once our network starts learning features, it is required to have some layers to merge learnt features and make some more sense out of it and hence we must use MaxPooling
  * MaxPooling placement
    * From Input Layer - Better to have it after few layers (2-3), so that network learns enough simple features and complicated features
    * From Prediction Layer - Better to have it before few layers (2-3), so that network accumulates all the learnt features and helps in final prediction
  * 1x1 Convolutions - This is very useful in reducing z-axis of the kernel. And alongside MaxPooling it helps merging common features and learn complicated features out of already learnt features
  * Convolution Block - It is better to design the network in blocks. We must have a bunch of layers with increasing number of channels so that the network learns lot of features. And this forms the Convolution Block
  * Transition Layer - MaxPooling and 1x1 Convolution forms the Transition block as it reduces the image size in all the three axis. This forms a very useful layer for reducing the number of parameters and for learning complex features by merging simple features
  * Positioning of Convolution Block and Transition Layer - Having placed them one after the other, makes  the network learn better. Learns features, merges them and learns more about. Then again learns some more features and again merges them to learn more about it
  * Number of Paramters - Depending on the compute requirements, this number must be kept in check. Lower the number of paramters, faster the network. And hence network must be designed by keeping this in mind
  * Receptive Field - It is better to keep track of Receptive field after each layer. We must look at the dataset and see what will be the resolution after which there is not much information learnt. For MNIST dataset and many other datasets, it is better to stop after 11x11 Receptive field, because after that we don't learn much from the data
  * Always add output layer size & receptive field info along side network layers, helps a lot in designing the network
  * SoftMax - Should be added at our prediction layer as it gives clear confident output. It increases the difference between the output values

* Once above details are decided, we must run the network and see how it performs. Its okay to not have the best accuracy, but it should be going in that direction. So that we can later improve more on it
* This forms our base Architecture

#### Running the network
* Better to run it once to see how the current network performs, so that we can know what sort improvements should be done to it
* Epochs and Batch-Size should be set to right value depending on the compute available
* The time taken per epoch depends on the batch size. Higher the batch size, lesser the time per epoch
* Because we want backprop to learn from not just one class but from all the classes and hence we send images in batches based on random distribution
* Figuring out what params works for you:
  * First train with high number of params & get accuracy scores
  * Then reduce it, if no drop in accuracy, then reduce further. Stop when reduction is seen. And then choose your network based on accuracy score

#### Improving our base Architecture
* Batch Normalization - This takes care of normalizing the output of the convolution layers. And hence helps in achieving better accuracy. It is better to place this after every layer, expect before the prediction layer
* DropOut - Helps in reducing Overfitting i.e. the difference between training and validation accuracy
* Dropouts can be added after any layers, but shouldn't have it before prediction layer. Ideal value is the range of 0.1

#### Speeding up the Network
* Once we achieve better results. We can speed our network by figuring out the learning rate
* Learning rate (LR) changes based on batch size, higher the batch size, smaller the learning rate
* If LR is too high, then the network will struggle a lot with the minima
* If LR is too low, then network will take lot of time to reach the minima
* Hence LR must be set to right value. And right value is chosen based on Trial & Error method
