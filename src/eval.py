import os
import torch
import configs
import numpy as np
from dataset import CarDataset
from model import CarModelClassifier
from sklearn.metrics import confusion_matrix, classification_report


def eval_model():
    dataset = CarDataset(data_dir=configs.DATA_DIR)
    _, val = dataset.split()

    model = CarModelClassifier(output_size=len(dataset.classes))
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


if __name__ == "__main__":
    eval_model()