import base64
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg11

resize = transforms.Resize((224, 224), antialias=None)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
random = transforms.RandomResizedCrop(384, antialias=True)

model = vgg11()
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 1)

model.load_state_dict(torch.load("model.pt"))

original = np.array(Image.open("rick.png"))

st = input(f"Devin's image filtering service (Enter a base64 encoded image)\n")
try:
    img = np.frombuffer(base64.standard_b64decode(st), dtype=np.uint8)
    img = img.reshape(original.shape)
    assert np.max(np.abs(original.astype(np.int32) - img.astype(np.int32))) <= 8
except:
    print("Yet another peaceful day without rickroll")
    sys.exit(0)

image = Image.fromarray(img)
x = random(normalize(resize(to_tensor(image))).unsqueeze(0))
with torch.no_grad():
    y = torch.sigmoid(model(x))

if y > 0.175:
    print("Oops that's a rickroll")
    sys.exit(0)

print("Arrrgghhh got rickrolled")
with open("flag.txt") as f:
    print(f.read())