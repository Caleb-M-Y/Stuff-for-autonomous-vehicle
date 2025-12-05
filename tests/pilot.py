import sys
from pathlib import Path
from torchinfo import summary

sys.path.append(str(Path(__file__).parents[1].joinpath("scripts", "cnn_architectures")))
print(sys.path)  # debug
from bear_net import BearNet


model = BearNet()  # Adjust num_classes as needed
batch_size = 1
summary(model, input_size=(batch_size, 3, 224, 224))
