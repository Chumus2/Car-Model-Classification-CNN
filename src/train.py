import configs
from dataset import CarDataset
from colorama import Fore, Style, init
from model import CarModelClassifier


init(autoreset=True)

def train_model():
    dataset = CarDataset(data_dir=configs.DATA_DIR, transform=configs.TRANSFORMS)
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Dataset classes: {dataset.classes}")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Dataset size: {len(dataset)}")


if __name__ == "__main__":
    train_model()