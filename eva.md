### AI Notes from Extensive Vision AI Program - 2019

#### When to use strides > 1?

* When classification is not that important. Like counting objects on a conveyor belt
* To make network faster
* Suitable for low end hardware like Raspberry PI
* Using strides > 1 causes issues like checkerboard. Network sees the image in blurred fashion like how one having spectacles see the world without spectacles

#### Common network building blocks

* For a input size of 400x400x3, we usually convolve with 3x3 kernel till receptive field of 11x11 is reached with increasing number of channels: 32 -> 64 -> 128 -> 256 -> 512. This forms one **convolution block**
* Usually till here edges & gradients are learnt by the kernels
* 1x1 will merge features rather than figuring out something new like what 3x3 kernel does. It merges edges & gradients seen together in spatial domain and not form new textures out of it. 3x3 can achieve the same thing, but for that it has to reach a receptive field to see all the common edges in the spatial domain
* 1x1 also acts like a filter. Say in a imageNet network, a image of dog with sofa & bed in background in it. 1x1 will filter sofa & bed from it only passing dog in further layers
* 1x1 reduces number of channels
* 1x1 with Max-Pooling forms **transition block**. Because it reduces image dimensions without loosing much information
* This is how modern networks are designed:
  * Input Layer
  * Convolution Layer
  * Convolution Block - Transition Block
  * Convolution Block - Transition Block
  * . . .
  * Output Layer
* Above is called `Squeeze & Excitation Architecture`

#### Methodology for building a network

* Look at images in the dataset and figure out the final receptive field which will cover all the images. Also make sure there is no much data on the edges, as convolution without padding ignores data at edges
* Perform Image Augmentation by first looking at the dataset
* Build the network with above convolution & transition blocks
* Network should be designed to gather following information:
  * Edges & Gradients
  * Textures
  * Patterns
  * Part Of Objects
  * Objects
* Max pooling can be used after each information is found
* Do not use Fully Connected Layers
* Use `ReLU` as activation function
* Do not use Activation function like ReLU before Softmax, as negative values are required for Softmax calculations

#### What is more important RAM or CPU cores?

* RAM. Because network will be loaded in RAM. If we have less RAM and more CPU, then we can't load the network and hence no use of CPU cores

#### How to think about Architecture

* First don't aim at reducing the number of parameters. Aim at getting the network architecture right
* Always add output layer size & receptive field info along side network layers
* Since we are not using Global Average Pooling (as of now), we are using kernel of same size as input to the final layer to get single output
* When we reach 11x11 or 10x10, we have almost lost our data so no point convolving beyond that (assumption is for MNIST dataset)
* Bias is of no use for Image Convolution, because Kernel are never zero. Hence we always set bias to 0
* But in the calculation of params, bias comes into picture. 1 per kernel. If 3x3 kernel & 32 of those, then params = [(3x3)+1]x32 = 320 params
* The time taken per epoch depends on the batch size. Higher the batch size, lesser the time per epoch
* Because we want backprop to learn from not just one class but from all the classes and hence we send images in batches based on random distribution
* Learning rate changes based on batch size, higher the batch size, smaller the learning rate
* Figuring out what params works for you:
  * First train with high number of params & get accuracy scores
  * Then reduce it, if no drop in accuracy, then reduce further. Stop when reduction is seen. And then choose your network based on accuracy score
* Once network is fixed, then add thinbgs like Batch Normalization, Dropout etc.
* Should RELU come before BN or viceversa? -> No clear answer as of now
* Dropouts can be added after any layers, but shouldn't have it before prediction layer. Ideal value is the range of 0.1
* Sample network design flow:
  * 1st DNN:
    * Simple squeeze & excite network, no BN, Dropout, ImgAug, LRScheduler etc
    * High number of parameters
  * 2nd DNN:
    * Reduce number of channels in each layer and hence reduction in number of parameters
    * Same Architecture is maintained
  * 3rd DNN:
    * Add BN layers
  * 4th DNN:
    * Increase some more parameters as we wanted it to under 15K, hence we have the luxury to increase it
  * 5th DNN:
    * Add Dropout
  * 6th DNN:
    * Set Learning Rate, now with less epochs we get our accuracy

# Final Q&A Session 8
* With Depthwise separable convolution -> less params, high kernel size -> High expressivity with low cost
* Today nobody uses Add, only Concat
* Upsampling doesn't help with image accuracy. It is only used for super resolution & encoder/decoder networks
* For concat of different output sizes, we must use `space-to-depth`
* Reducing learning rate is also regularization
* Droput in Fully Convolution Layers, drops pixels -> which is not a good strategy for DNN
* Use spatial dropouts, which drops channels
* With increased padding, we loose information
* We avoid stride > 1, as it adds checkerboard. But if we want to run it on constrained hw, then we do use it.
* For MNIST our base batch size can be 128, For CIFAR10 256 or 512. Once we get some good accuracy with our set batch size, then play with LR to see if we can have some more improvements
* It is good to have bottleneck layers in your network. Add it avoids overfitting and does give better learning
* Resnet 34 -> ideal network to start with, as it gives good performance with low params
* Main focus should be on data augmentation & loss function
* For production deployment, we are left with Resnet | Inception | Densenet. Resnet == iOS user, Inception == Android User, Densenet == Blackberry user
* Normalization should also be done on testing dataset as well
* LR will depend on which stage we are, hence better to use a LR calculator which will figure out LR for use
