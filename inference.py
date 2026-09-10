import torch
from PIL import Image
import torchvision.transforms as transforms

from model import CNN


classes = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def load_model(weights_path="weights/cifar10_small_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    return model, device


def predict_scores(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(image)

    return scores[0].cpu()


# Example:
# model, device = load_model()
# scores = predict_scores("image.png", model, device)
# print(scores)
# print("Predicted class:", classes[scores.argmax().item()])
