import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from inference import get_transform, load_model


BATCH_SIZE = 128


def evaluate(weights_path: str = "weights/cifar10_small_cnn.pth") -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device=device)

    testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=get_transform(),
    )
    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    evaluate()
