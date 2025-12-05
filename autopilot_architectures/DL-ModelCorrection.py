import torch.nn as nn

# input shape: (180, 200,3)
# based off modified ResNet architecture from assignment 6
# Using LeakyReLU activations instead of ReLU
# 
class DummyPilotNet(nn.Module):
    def __init__(self):
        super(DummyPilotNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2) # outptut: (32, 90, 100)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # output: (32, 45, 50)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # output: (64, 45, 50)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # output: (128, 23, 25)
        self.bn3 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # output: (128, 1, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.01)
        self.flatten = nn.Flatten()

    def forward(self, x):  # input shape: (180, 200, 3)
        x = self.leakyReLU(self.conv1(x))  # output: (32, 90, 100)
        # 180 - 5 + 2*2 /2 +1 = 90 | 200 - 5 + 2*2 /2 +1 = 100 
        # 32(5*5*3+1) = 2432 parameters
        x = self.bn1(x) # 32 * 2 = 64 parameters
        x = self.maxpool1(x)          # output: (32, 45, 50)
        # 90 / 2 = 45 | 100 / 2 = 50 
        x = self.leakyReLU(self.conv2(x))  # output: (64, 45, 50)
        # 45 - 3 + 2*1 /1 +1 = 45 | 50 - 3 + 2*1 /1 +1 = 50
        # 64(3*3*32+1) 18496 parameters
        x = self.bn2(x) # 64 * 2 = 128 parameters
        x = self.leakyReLU(self.conv3(x))  # output: (128, 23, 25)
        # 45 - 3 + 2*1 /2 +1 = 23 | 50 - 3 + 2*1 /2 +1 = 25
        # 128(3*3*64+1) = 73856 parameters
        x = self.bn3(x) # 128 * 2 = 256 parameters
        x = self.avgpool(x)           # output: (128, 1, 1)
        x = self.flatten(x)           # output: (128)
        x = self.dropout(x)
        x = self.leakyReLU(self.fc1(x))    # output: (64)
        # 64 * 128 + 64 = 8256 parameters
        y = self.fc2(x)               # output: (2)
        # 2 * 64 + 2 = 130 parameters
        # total parameters:  2432 + 64 + 18496 + 128 + 73856 + 256 + 8256 + 130 = 103618
        return y 