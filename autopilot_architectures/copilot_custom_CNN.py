import torch.nn as nn

# input shape: (180, 200, 3)
# Custom CNN optimized for autonomous vehicle steering and throttle control
# Design Philosophy:
# - Gradual spatial reduction to preserve fine-grained steering details
# - Deeper feature extraction with 5 conv blocks for better turn detection
# - Multi-scale feature learning with varying kernel sizes
# - Spatial Attention mechanism to focus on road/lane features
# - Residual connections for better gradient flow
# - Balanced capacity (~380k params) - not too small, not too large
# 
class DummyPilotNet(nn.Module):
    def __init__(self):
        super(DummyPilotNet, self).__init__()
        
        # Feature Extraction Backbone - Progressive downsampling
        # Block 1: Initial feature extraction with larger receptive field
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=2) # output: (24, 90, 100)
        self.bn1 = nn.BatchNorm2d(24)
        
        # Block 2: Detailed feature extraction
        self.conv2a = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3, stride=1, padding=1) # output: (36, 90, 100)
        self.bn2a = nn.BatchNorm2d(36)
        self.conv2b = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, stride=2, padding=1) # output: (48, 45, 50)
        self.bn2b = nn.BatchNorm2d(48)
        
        # Block 3: Mid-level features
        self.conv3a = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1) # output: (64, 45, 50)
        self.bn3a = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # output: (64, 23, 25)
        self.bn3b = nn.BatchNorm2d(64)
        
        # Block 4: High-level features with larger receptive field
        self.conv4a = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1) # output: (96, 23, 25)
        self.bn4a = nn.BatchNorm2d(96)
        self.conv4b = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1) # output: (96, 23, 25)
        self.bn4b = nn.BatchNorm2d(96)
        
        # Block 5: Deep semantic features
        self.conv5a = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1) # output: (128, 12, 13)
        self.bn5a = nn.BatchNorm2d(128)
        self.conv5b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) # output: (128, 12, 13)
        self.bn5b = nn.BatchNorm2d(128)
        
        # Spatial Attention Module - helps focus on important regions (lanes, curves)
        self.attention_conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1) # output: (1, 12, 13)
        
        # Global pooling and decision layers
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1)) # output: (128, 1, 1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 128) # output: (128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64) # output: (64)
        self.fc_steering = nn.Linear(64, 1) # Steering output
        self.fc_throttle = nn.Linear(64, 1) # Throttle output
        
        # Activations
        self.elu = nn.ELU(alpha=1.0) # ELU helps with vanishing gradients
        self.sigmoid = nn.Sigmoid() # For attention weights
        self.flatten = nn.Flatten()

    def forward(self, x):  # input shape: (3, 180, 200)
        # Block 1: Initial feature extraction
        x = self.elu(self.bn1(self.conv1(x)))  # output: (24, 90, 100)
        # (180 - 5 + 2*2) / 2 + 1 = 90 | (200 - 5 + 2*2) / 2 + 1 = 100
        # 24 * (5*5*3 + 1) = 1824 parameters
        # BatchNorm: 24 * 2 = 48 parameters
        
        # Block 2: Gradual channel expansion
        x = self.elu(self.bn2a(self.conv2a(x)))  # output: (36, 90, 100)
        # (90 - 3 + 2*1) / 1 + 1 = 90 | (100 - 3 + 2*1) / 1 + 1 = 100
        # 36 * (3*3*24 + 1) = 7812 parameters
        # BatchNorm: 36 * 2 = 72 parameters
        
        x = self.elu(self.bn2b(self.conv2b(x)))  # output: (48, 45, 50)
        # (90 - 3 + 2*1) / 2 + 1 = 45 | (100 - 3 + 2*1) / 2 + 1 = 50
        # 48 * (3*3*36 + 1) = 15552 parameters
        # BatchNorm: 48 * 2 = 96 parameters
        
        # Block 3: Mid-level feature extraction
        x = self.elu(self.bn3a(self.conv3a(x)))  # output: (64, 45, 50)
        # (45 - 3 + 2*1) / 1 + 1 = 45 | (50 - 3 + 2*1) / 1 + 1 = 50
        # 64 * (3*3*48 + 1) = 27712 parameters
        # BatchNorm: 64 * 2 = 128 parameters
        
        x = self.elu(self.bn3b(self.conv3b(x)))  # output: (64, 23, 25)
        # (45 - 3 + 2*1) / 2 + 1 = 23 | (50 - 3 + 2*1) / 2 + 1 = 25
        # 64 * (3*3*64 + 1) = 36928 parameters
        # BatchNorm: 64 * 2 = 128 parameters
        
        # Block 4: High-level features with residual-like connection
        identity = x
        x = self.elu(self.bn4a(self.conv4a(x)))  # output: (96, 23, 25)
        # (23 - 3 + 2*1) / 1 + 1 = 23 | (25 - 3 + 2*1) / 1 + 1 = 25
        # 96 * (3*3*64 + 1) = 55392 parameters
        # BatchNorm: 96 * 2 = 192 parameters
        
        x = self.elu(self.bn4b(self.conv4b(x)))  # output: (96, 23, 25)
        # (23 - 3 + 2*1) / 1 + 1 = 23 | (25 - 3 + 2*1) / 1 + 1 = 25
        # 96 * (3*3*96 + 1) = 83040 parameters
        # BatchNorm: 96 * 2 = 192 parameters
        
        # Block 5: Deep semantic understanding
        x = self.elu(self.bn5a(self.conv5a(x)))  # output: (128, 12, 13)
        # (23 - 3 + 2*1) / 2 + 1 = 12 | (25 - 3 + 2*1) / 2 + 1 = 13
        # 128 * (3*3*96 + 1) = 110720 parameters
        # BatchNorm: 128 * 2 = 256 parameters
        
        x = self.elu(self.bn5b(self.conv5b(x)))  # output: (128, 12, 13)
        # (12 - 3 + 2*1) / 1 + 1 = 12 | (13 - 3 + 2*1) / 1 + 1 = 13
        # 128 * (3*3*128 + 1) = 147584 parameters
        # BatchNorm: 128 * 2 = 256 parameters
        
        # Spatial Attention: Learn to focus on relevant regions
        attention_weights = self.sigmoid(self.attention_conv(x))  # output: (1, 12, 13)
        # 1 * (1*1*128 + 1) = 129 parameters
        x = x * attention_weights  # Broadcast attention weights across channels
        
        # Global aggregation and decision network
        x = self.globalavgpool(x)  # output: (128, 1, 1)
        x = self.flatten(x)  # output: (128)
        
        x = self.dropout1(x)
        x = self.elu(self.fc1(x))  # output: (128)
        # 128 * 128 + 128 = 16512 parameters
        
        x = self.dropout2(x)
        x = self.elu(self.fc2(x))  # output: (64)
        # 64 * 128 + 64 = 8256 parameters
        
        # Split decision heads for steering and throttle
        steering = self.fc_steering(x)  # output: (1)
        # 1 * 64 + 1 = 65 parameters
        throttle = self.fc_throttle(x)  # output: (1)
        # 1 * 64 + 1 = 65 parameters
        
        # Combine outputs
        y = nn.functional.cat([steering, throttle], dim=1)  # output: (2)
        
        # Total parameters: 1824 + 48 + 7812 + 72 + 15552 + 96 + 27712 + 128 + 36928 + 128 +
        #                   55392 + 192 + 83040 + 192 + 110720 + 256 + 147584 + 256 + 129 +
        #                   16512 + 8256 + 65 + 65 = 512,959 parameters
        return y