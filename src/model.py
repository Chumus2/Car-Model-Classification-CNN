import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CarModelClassifier(nn.Module):
    def __init__(self, output_size, img_size=(256, 256), hidden_size=128):
        super().__init__()

        self.features = nn.Sequential(
            # 3 > 32
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 32 > 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 64 > 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128 * 4 * 4, out_features=hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CarModelClassifierTL(nn.Module):
    def __init__(self, output_size, pretrained=True, freeze_backbone=False):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features=in_features, out_features=output_size),
        )

    def forward(self, x):
        return self.backbone(x)