import torch.nn as nn


class DummyPilotNet(nn.Module):
    def __init__(self):
        super(DummyPilot, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(180 * 200 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    from torchinfo import summary

    model = DummyPilotNet()  # Adjust num_classes as needed
    summary(model, input_size=(1, 3, 180, 200))
