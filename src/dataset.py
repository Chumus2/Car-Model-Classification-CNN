import copy
import os
import configs
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


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
        val_size = len(self.samples) - train_size

        train_dataset, val_dataset = random_split(self, [train_size, val_size])

        self.transform = configs.TRAIN_TRANSFORMS

        val_dataset.dataset = copy.deepcopy(self)
        val_dataset.dataset.transform = configs.VAL_TRANSFORMS

        train = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
        val = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

        return train, val