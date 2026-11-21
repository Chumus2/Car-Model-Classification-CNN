import os
import torch
import configs
import numpy as np
from glob import glob
from PIL import Image
from dataset import CarDataset
from colorama import Fore, Style, init
from model import CarModelClassifier, CarModelClassifierTL
from sklearn.metrics import confusion_matrix, classification_report


init(autoreset=True)

def eval_model():
    dataset = CarDataset(data_dir=configs.DATA_DIR)
    _, val = dataset.split()

    model = CarModelClassifierTL(output_size=len(dataset.classes))
    model = model.to(configs.DEVICE)

    checkpoint_path = os.path.join(configs.MODEL_DIR, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=configs.DEVICE, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val:
            images = images.to(configs.DEVICE)

            outputs = model(images)
            _, pred = torch.max(outputs, 1)

            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    classes = dataset.classes

    print("=" * 50)
    print("REPORT BY CLASSES")
    print("=" * 50 + "\n")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=2, zero_division=0))

    mistakes = []
    for true_idx in range(len(classes)):
        for pred_idx in range(len(classes)):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                mistakes.append((cm[true_idx, pred_idx], classes[true_idx], classes[pred_idx]))

    mistakes.sort(reverse=True)

    print("\n" + "=" * 50)
    print("TOP MISTAKES (TRUE => PREDICTED)")
    print("=" * 50 + "\n")

    for count, true_cls, pred_cls in mistakes[:10]:
        print(f"{true_cls} => {pred_cls}: {count} times")


def test_model():
    print(f"\n{Fore.YELLOW}[INFO]{Style.RESET_ALL} Working device: {configs.DEVICE}")

    test_images = []

    for ext in configs.IMG_EXT:
        pattern = f"*{ext}" if not ext.startswith("*") else ext
        files = glob(os.path.join(configs.TEST_DIR, "**", pattern), recursive=True)
        test_images.extend(files)

    test_images = sorted([
        p for p in set(test_images)
        if os.path.splitext(p)[1].lower() in configs.IMG_EXT
    ])
    print(f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Test images found: {len(test_images)}")

    model_path = os.path.join(configs.MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Checkpoint not found at {model_path}")
        return

    if not test_images:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No test images found in {configs.TEST_DIR}.")
        return

    checkpoint = torch.load(model_path, map_location=configs.DEVICE, weights_only=False)
    classes = checkpoint.get("classes", configs.CLASSES)

    model = CarModelClassifierTL(output_size=len(classes))
    model = model.to(configs.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\n" + "=" * 75)
    print("BATCH INFERENCE (PREDICTION TEST)")
    print("=" * 75)

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"{Fore.RED}[Error]{Style.RESET_ALL} Image not found at {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        input_tensor = configs.VAL_TRANSFORMS(img).unsqueeze(0).to(configs.DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        pred_label = classes[predicted.item()]
        conf_score = confidence.item() * 100

        print(f"File: {os.path.basename(img_path)} > Prediction: "
              f"{Fore.GREEN}{pred_label}{Style.RESET_ALL} "
              f"(Confidence: {conf_score:.2f}%)"
        )
        print("=" * 75)


if __name__ == "__main__":
    eval_model()
    test_model()