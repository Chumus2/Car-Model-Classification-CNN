import configs
from dataset import CarDataset
from model import CarModelClassifier
from colorama import Fore, Style, init


init(autoreset=True)

def train_model():
    dataset = CarDataset(data_dir=configs.DATA_DIR, transform=configs.TRANSFORMS)
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Dataset classes: {dataset.classes}")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Dataset size: {len(dataset)}")

    train, val = dataset.split()
    print(f"\n{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Dataset splitted!")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Train size: {len(train)}")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Val size: {len(val)}")


if __name__ == "__main__":
    train_model()