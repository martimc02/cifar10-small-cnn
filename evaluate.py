import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import SmallCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

model = SmallCNN().to(device)
model.load_state_dict(torch.load("weights/cifar10_small_cnn.pth", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test images: {100 * correct / total:.2f}%")
