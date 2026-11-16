import torch
import configs
import torch.nn as nn
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

    model = CarModelClassifier(output_size=len(dataset.classes))
    model = model.to(configs.DEVICE)
    print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} Working device: {configs.DEVICE}")

    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)
    print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} Loss: Cross Entropy Loss")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Optimizer: Adam (lr={configs.LEARNING_RATE})")

    print("\n" + "=" * 50)
    print("STARTING TRAINING PROCESS")
    print("=" * 50 + "\n")
    for epoch in range(configs.EPOCHS):

        model.train()
        train_loss = 0.0

        for images, labels in train:
            images = images.to(configs.DEVICE)
            labels = labels.to(configs.DEVICE)

            optim.zero_grad()
            outputs = model(images)
            loss = crit(outputs, labels)
            loss.backward()
            optim.step()

            train_loss += loss.item()

        train_avg_loss = train_loss / len(train)
        print(f"Train_avg_loss: {train_avg_loss}")


if __name__ == "__main__":
    train_model()