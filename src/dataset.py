import copy
import os
import torch
import configs
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset


class CarDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform

        self.classes = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(self.data_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue

            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(configs.IMG_EXT):
                    img_path = os.path.join(cls_folder, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def split(self):
        train_size = int(len(self.samples) * configs.TRAIN_SIZE)

        generator = torch.Generator().manual_seed(configs.SEED)
        indices = torch.randperm(len(self.samples), generator=generator).tolist()

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_base = CarDataset(self.data_dir, transform=configs.TRAIN_TRANSFORMS)
        val_base = CarDataset(self.data_dir, transform=configs.VAL_TRANSFORMS)

        train_base.classes, train_base.class_to_idx, train_base.samples = self.classes, self.class_to_idx, self.samples
        val_base.classes, val_base.class_to_idx, val_base.samples = self.classes, self.class_to_idx, self.samples

        train_dataset = Subset(train_base, train_indices)
        val_dataset = Subset(val_base, val_indices)

        train = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
        val = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

        return train, val