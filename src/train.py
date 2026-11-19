import os.path

import torch
import configs
import torch.nn as nn
from dataset import CarDataset
from model import CarModelClassifier
from colorama import Fore, Style, init
from torch.optim.lr_scheduler import ReduceLROnPlateau


init(autoreset=True)

def train_model():
    dataset = CarDataset(data_dir=configs.DATA_DIR)
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Dataset classes: {dataset.classes}")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Dataset size: {len(dataset)}")

    train, val = dataset.split()
    print(f"\n{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Dataset splitted!")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Train size: {len(train.dataset)}")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Val size: {len(val.dataset)}")

    model = CarModelClassifier(output_size=len(dataset.classes))
    model = model.to(configs.DEVICE)
    print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} Working device: {configs.DEVICE}")

    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=configs.FACTOR, patience=configs.PATIENCE)
    print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} Loss: Cross Entropy Loss")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Optimizer: Adam (lr={configs.LEARNING_RATE})")
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Scheduler: ReduceLROnPlateau")

    best_val_loss = float("inf")
    patience_count = 0

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


        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():

            for images, labels in val:
                images = images.to(configs.DEVICE)
                labels = labels.to(configs.DEVICE)

                outputs = model(images)
                loss = crit(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_avg_loss = val_loss / len(val)
        val_acc = correct / total * 100

        scheduler.step(val_avg_loss)
        current_lr = optim.param_groups[0]["lr"]

        print(f"Epoch [{epoch + 1}/{configs.EPOCHS}] "
              f"| Train Loss: {train_avg_loss:.4f} "
              f"| {Fore.GREEN}Val Loss: {val_avg_loss:.4f} {Style.RESET_ALL} "
              f"| {Fore.GREEN}Val Accuracy: {val_acc:.2f}% {Style.RESET_ALL} "
              f"| LR: {current_lr}")

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_count = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "classes": dataset.classes,
                "train_loss": train_avg_loss,
                "val_loss": val_avg_loss,
                "val_acc": val_acc,
            }

            if not os.path.exists(configs.MODEL_DIR):
                os.makedirs(configs.MODEL_DIR)

            checkpoint_path = os.path.join(configs.MODEL_DIR, "best_model.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"{Fore.GREEN}[CHECKPOINT]{Style.RESET_ALL} Improvement found! "
                  f"Saved new best_model.pth (Val Loss: {best_val_loss:.4f})\n")

        else:
            patience_count += 1
            if patience_count >= configs.PATIENCE:
                print(f"{Fore.RED}[STOP]{Style.RESET_ALL} Validation loss did not improve for {configs.PATIENCE} consecutive epochs.")
                break
        

if __name__ == "__main__":
    train_model()