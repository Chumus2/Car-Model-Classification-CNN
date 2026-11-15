import torch.nn as nn


class CarModelClassifier(nn.Module):
    def __init__(self, input_size, output_size, img_size, hidden_size=128, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=padding)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            in_features=64 * (img_size[0] // 4) * (img_size[1] // 4),
            out_features=hidden_size
        )
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x