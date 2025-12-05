"""
If logged in remotely, please enable X11 forwarding, either `ssh -X` or `ssh -Y`
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.transforms import v2

# SETUP
# Prepare torchvision transforms
to_tensor = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# LOOP
raw_im = np.load("image_array.npy")
imt = to_tensor(raw_im)
imp = F.to_pil_image(imt)
ima = np.asarray(imp)
plt.imshow(ima)
