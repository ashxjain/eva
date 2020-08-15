# Architectural Basics

* We'll start with a base network and see how we improve it and there by learning on how to use different methologies to achieve desired results.

* Base network architecutre:
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28x28x1 -> 28x28x32 [Jin=1,K=3,RFin=1] RF: 3, Jout: 1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # -> 28x28x64 [Jin=1,K=3,RFin=3] RF: 5, Jout: 1
        self.pool1 = nn.MaxPool2d(2, 2) # -> 14x14x64 [Jin=1,K=2,RFin=5] RF: 6, Jout: 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # -> 14x14x128 [Jin=2,K=3,RFin=6] RF: 10, Jout: 2
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # -> 14x14x256 [Jin=2,K=3,RFin=10] RF: 14, Jout: 2
        self.pool2 = nn.MaxPool2d(2, 2) # -> 7x7x256 [Jin=2,K=2,RFin=14] RF: 16, Jout: 4
        self.conv5 = nn.Conv2d(256, 512, 3) # -> 5x5x512 [Jin=4,K=3,RFin=16] RF: 24, Jout: 4
        self.conv6 = nn.Conv2d(512, 1024, 3) # -> 3x3x1024 [Jin=4,K=3,RFin=24] RF: 32, Jout: 4
        self.conv7 = nn.Conv2d(1024, 10, 3) # -> 1x1x10 [Jin=4,K=3,RFin=32] RF: 40, Jout: 4

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)

# Number of Parameters = 6.3M
# Batch Size = 128
# 1st epoch Acc = 29%
```

* As seen above there are lots of parameters for such a simple dataset
* RELU after last layer is a bad thing to do. On removing RELU, on the same network we achieve 98%!
* Tried the following to achieve very good results:
  1. Removed RELU after last layer (reached 98% in first epoch)
  2. Modified batch size to 64. But no improvement, so reverted back to 128
  3. Reduced number of parameters to less than 20K
  4. Refactored network to use `nn.Sequential`, makes code more readable. Also added functions to avoid code duplication
  5. Added transition block, this makes our architecture `excite and squeeze` network. This makes sense so that network can learn Edge/Gradients/Textures/Pattern/PartsOfObject during excite phase and make sense of it during squeeze phase. Purposefully added bunch of convolution layers after last transition block so that MaxPooling is little far from Prediction layer.
  6. Added Batch Normalization after every conv layer, except last layer
  7. With above things, i was able to achieve 99.36% Accuracy at 18th Epoch. Training loss was 0.0036, but test loss was 0.0218. This was clearly overfitting! So tried two things to fix it: Dropout & Image Preprocessing
     1. Tried Dropout with values 0.1,0.2,etc but it was not performing well
     2. Tried adding image preprocesing i.e. RandomAffine/ColorJitter. Worked very well! Was able to achieve 99.45% Accuracy at 9th Epoch. And 99.50 at 20th Epoch.
* My goal was to achieve 99.4% accurach in less than 20 epochs with less than 20K paramters, which i achieved. But there are lot more things we can try to improve the network. For example:
  1. Better learning rate
  2. Try different batch size