import os
import configs
from PIL import Image
from torch.utils.data import Dataset


class CarDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform

        self.classes = sorted(os.listdir(data_dir))
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
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label