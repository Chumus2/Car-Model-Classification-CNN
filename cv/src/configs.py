import os
import torch
from torchvision import transforms

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image / Transforms
IMG_EXT = (".jpg", ".jpeg", ".png")
IMG_WIDTH, IMG_HEIGHT = 256, 256
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
    ),
    transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dirs
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
TEST_DIR = os.path.join(ROOT_DIR, "test")

# Data
SEED = 42
TRAIN_SIZE = 0.8
BATCH_SIZE = 32
CLASSES = [
    "Audi R8", "Audi RS5", "BMW M3", "BMW M5", "Chevrolet Camaro", "Ford Mustang",
    "Mercedes AMG GT", "Nissan GT R", "Porsche 911", "Toyota Supra"
]

# Training
EPOCHS = 50
FACTOR = 0.5
PATIENCE = 10
SCHEDULER_PATIENCE = 3
LEARNING_RATE = 0.001