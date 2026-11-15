import os
import torch
from torchvision import transforms

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image / Transforms
IMG_EXT = (".jpg", ".jpeg", ".png")
IMG_WIDTH, IMG_HEIGHT = 256, 256
TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
])

# Dirs
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Data
TRAIN_SIZE = 0.8
BATCH_SIZE = 32
LEARNING_RATE = 0.001