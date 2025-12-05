import torch.nn as nn

# input shape: (180, 200, 3)
# based off modified ResNet architecture from assignment 6
# Using LeakyReLU activations instead of ReLU
# Removed MaxPool to preserve spatial information
# 
class DummyPilotNet(nn.Module):
    def __init__(self):
        super(DummyPilotNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2) # output: (32, 90, 100)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1) # output: (64, 45, 50)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # output: (128, 23, 25)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) # output: (128, 23, 25)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1) # output: (128, 12, 13)
        
        # Global pooling and fully connected layers
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1)) # output: (128, 1, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 2)
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.01)
        self.flatten = nn.Flatten()

    def forward(self, x):  # input shape: (3, 180, 200)
        x = self.leakyReLU(self.conv1(x))  # output: (32, 90, 100)
        # (180 - 5 + 2*2) / 2 + 1 = 90 | (200 - 5 + 2*2) / 2 + 1 = 100 
        # 32 * (5*5*3 + 1) = 2432 parameters
        
        x = self.leakyReLU(self.conv2(x))  # output: (64, 45, 50)
        # (90 - 3 + 2*1) / 2 + 1 = 45 | (100 - 3 + 2*1) / 2 + 1 = 50
        # 64 * (3*3*32 + 1) = 18496 parameters
        
        x = self.leakyReLU(self.conv3(x))  # output: (128, 23, 25)
        # (45 - 3 + 2*1) / 2 + 1 = 23 | (50 - 3 + 2*1) / 2 + 1 = 25
        # 128 * (3*3*64 + 1) = 73856 parameters
        
        x = self.leakyReLU(self.conv4(x))  # output: (128, 23, 25)
        # (23 - 3 + 2*1) / 1 + 1 = 23 | (25 - 3 + 2*1) / 1 + 1 = 25
        # 128 * (3*3*128 + 1) = 147584 parameters

        x = self.leakyReLU(self.conv5(x))  # output: (128, 12, 13)
        # (23 - 3 + 2*1) / 2 + 1 = 12 | (25 - 3 + 2*1) / 2 + 1 = 13
        # 128 * (3*3*128 + 1) = 147584 parameters
        
        x = self.globalavgpool(x)     # output: (128, 1, 1)
        x = self.flatten(x)           # output: (128)
        x = self.dropout(x)
        x = self.leakyReLU(self.fc1(x))    # output: (128)
        # 128 * 128 + 128 = 16512 parameters
        y = self.fc2(x)               # output: (2)
        # 2 * 128 + 2 = 258 parameters
        
        # Total parameters: 2432 + 18496 + 73856 + 147584 + 147584 + 16512 + 258 = 406722 parameters
        return y