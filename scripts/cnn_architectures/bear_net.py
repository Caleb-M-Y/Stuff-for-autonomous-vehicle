import torch.nn as nn


class BearNet(nn.Module):
    def __init__(self):
        super(BearNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):  # 224
        x = self.relu(self.conv1(x))  # (224 - 7 + 2 * 3) / 2 + 1 = 112.5
        x = self.max_pool(x)  # (112 - 3 + 2 * 1) / 2 + 1 = 56.5
        x = self.relu(self.conv2(x))  # (56 - 3) + 1 = 54
        x = self.relu(self.conv3(x))  # (54 - 3) / 2 + 1 = 26.5
        x = self.relu(self.conv4(x))  # (26 - 3) + 1 = 24
        x = self.relu(self.conv5(x))  # (24 - 3) / 2 + 1 = 11.5
        x = self.relu(self.conv6(x))  # (11 - 3) + 1 = 9
        x = self.relu(self.conv7(x))  # (9 - 3) + 1 = 7

        x = x.view(x.size(0), -1)  # flatten

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y = self.fc3(x)
        return y


if __name__ == "__main__":
    from torchinfo import summary

    model = BearNet()  # Adjust num_classes as needed
    batch_size = 1
    summary(model, input_size=(batch_size, 3, 224, 224))
