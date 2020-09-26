import torch.nn as nn
import torch.nn.functional as F

class QuizDNN(nn.Module):

    def conv_block (self, in_channels, out_channels, kernel_size = 3, padding = 1):
      return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.01))

    def __init__(self, opts=[]):
        super(QuizDNN, self).__init__()
        self.input = self.conv_block(3, 32)
        self.conv1 = self.conv_block(32, 32)
        self.conv2 = self.conv_block(32, 32)
        self.conv3 = self.conv_block(32, 32)
        self.conv4 = self.conv_block(32, 32)
        self.conv5 = self.conv_block(32, 32)
        self.conv6 = self.conv_block(32, 32)
        self.conv7 = self.conv_block(32, 32)
        self.conv8 = self.conv_block(32, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=8))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x1 = self.input(x) # 32 x 32 x 32
        x2 = self.conv1(x1) # 32x32x32
        #print(x1.shape, x2.shape)
        x3 = self.conv2(x1 + x2) #32x32x32
        x4 = self.pool(x1 + x2 + x3) #16x16x32
        x5 = self.conv3(x4) #16x16x32
        x6 = self.conv4(x4 + x5) #16x16x32
        x7 = self.conv5(x4 + x5 + x6) #16x16x32
        x8 = self.pool(x5 + x6 + x7) #8x8x32
        x9 = self.conv6(x8) #8x8x32
        x10 = self.conv7(x8 + x9) #8x8x32
        x11 = self.conv8(x8 + x9 + x10) #8x8x32
        x12 = self.gap(x11) #1x1x32
        x13 = self.fc(x12.view(x12.size(0), -1)) #1x1x10
        x = x13.view(-1, 10)
        return F.log_softmax(x, dim=-1)
