### MNIST Classification using L1/L2/BatchNorm/GhostBatchNorm techniques

* **Assignment 6** by Ashish Jain & Rahul Jain

* Base Model: This is the base model we started with. It is a simple model where we used BatchNorm & Dropout & Image Augmentation techniques. With following configuration:

  * Batch Size = 128
  * Epochs = 15

* Following is the result of base model:

  * Parameters: 6,765

  * Best Test Accuracy: 99.52

  * Last 5 epochs:

    | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
    | ---------- | -------------- | --------- | ------------- |
    | 0.1435     | 98.14          | 0.0195    | 99.43         |
    | 0.1070     | 98.27          | 0.0192    | 99.42         |
    | 0.0379     | 98.19          | 0.0182    | 99.51         |
    | 0.0163     | 98.27          | 0.0181    | 99.52         |
    | 0.0192     | 98.30          | 0.0192    | 99.48         |


### Experiments with L1, L2, BatchNorm, GhostBatchNorm techniques

Tried the following techniques on above base model with same configuration as above except the following is run for 15 Epochs:

* Base Model with L1 + BN:
  * With regularization parameter (lambda) set to 0.001
  * Last 5 epoch stats:
    | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
    | ---------- | -------------- | --------- | ------------- |
    | 0.0835 | 98.44 | 0.0183 | 99.45 |
    | 0.0915 | 98.46 | 0.0174 | 99.47 |
    | 0.1342 | 98.42 | 0.0175 | 99.47 |
    | 0.1245 | 98.41 | 0.0175 | 99.54 |
    | 0.0845 | 98.39 | 0.0179 | 99.43 |
  * Adding L1 regularization didn't affect the result. It is almost same as what we got for base model
  
* Base Model with L2 + BN:
  
  * For L2, have setting the following args to optimizer SGD:
    ` optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)`
  * Last 5 epoch stats:
    | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
    | ---------- | -------------- | --------- | ------------- |
    | 0.0626 | 98.36 | 0.0188 | 99.40 |
    | 0.1241 | 98.41 | 0.0184 | 99.43 |
    | 0.0592 | 98.42 | 0.0194 | 99.38 |
    | 0.0543 | 98.39 | 0.0190 | 99.37 |
    | 0.0811 | 98.39 | 0.0187 | 99.41 |
  
  * Accuracy reduced slightly with L2 regularization
  * L2 regularization only helps if the dataset is complex, in this case it is not
  
* Base Model with L1 and L2 with BN
  
  * Same configuration is set as done for above L1/L2
    
  * Last 5 epoch stats:
    | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
    | ---------- | -------------- | --------- | ------------- |
    | 0.0801 | 98.34 | 0.0185 | 99.40 |
    | 0.1296 | 98.39 | 0.0185 | 99.39 |
    | 0.1104 | 98.47 | 0.0178 | 99.35 |
    | 0.1515 | 98.36 | 0.0191 | 99.34 |
    | 0.1539 | 98.42 | 0.0180 | 99.36 |
  
  * Again no improvement, this is because of L2 regularization
  
* Base Model with GBN
  
  * Num_splits set to 2
    
  * Last 5 epoch stats:
    | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
    | ---------- | -------------- | --------- | ------------- |
    | 0.0196 | 98.22 | 0.0206 | 99.33 |
    | 0.0659 | 98.17 | 0.0214 | 99.36 |
    | 0.1076 | 98.13 | 0.0220 | 99.35 |
    | 0.0562 | 98.25 | 0.0214 | 99.36 |
    | 0.0924 | 98.22 | 0.0211 | 99.38 |
  * Although accuracy is less compared to L1. But, there is scope for improvement if we run it for more number of epochs (as train accuracy is less)
  
* Base Model with L1 and L2 with GBN
  * Same configuration is set as done for above L1/L2/GBN
    
  * Last 5 epoch stats:
    
    | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
    | ---------- | -------------- | --------- | ------------- |
    | 0.2046 | 98.38 | 0.0168 | 99.38 |
    | 0.1108 | 98.38 | 0.0176 | 99.42 |
    | 0.0901 | 98.26 | 0.0184 | 99.34 |
    | 0.2001 | 98.41 | 0.0176 | 99.42 |
    | 0.0718 | 98.32 | 0.0177 | 99.37 |
    
  * L2 is causing the problem here. L1 gives some improvement, but again GBN reduces train accuracy and hence gives room for more improvement by running it for some more epochs

### Final Results Visualized

* Loss/Accuracy graphs of above experiments

![losses-accuracies](https://raw.githubusercontent.com/ashxjain/eva/master/P5_S6/images/graph_loss_acc.png)

* 25 mis-classified images with GhostBatchNorm. Most of the following images, will be predicted wrong by humans itself

![wrong-images](https://github.com/ashxjain/eva/blob/master/P5_S6/images/25misclassified_img_GBN.png?raw=true)
