from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN


BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHTS_PATH = Path("weights/cifar10_small_cnn.pth")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def train() -> SmallCNN:
    torch.manual_seed(0)
    device = get_device()
    print(f"Using device: {device}")

    trainset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=get_transform(),
    )
    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
    )

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for batch_index, (inputs, labels) in enumerate(trainloader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_index % 2000 == 0:
                print(
                    f"Epoch {epoch + 1}, batch {batch_index}: "
                    f"loss = {running_loss / 2000:.3f}"
                )
                running_loss = 0.0

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Finished training. Weights saved to {WEIGHTS_PATH}")
    return model


if __name__ == "__main__":
    train()
