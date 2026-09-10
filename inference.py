from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms

from model import SmallCNN


ImageInput = Union[str, Path, Image.Image, torch.Tensor]

CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_model(
    weights_path: Union[str, Path] = "weights/cifar10_small_cnn.pth",
    device: torch.device | None = None,
) -> SmallCNN:
    if device is None:
        device = get_device()

    model = SmallCNN().to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _prepare_image(image: ImageInput) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone().float()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError("A batched tensor must contain exactly one image.")
            tensor = tensor.squeeze(0)
        if tensor.shape != (3, 32, 32):
            raise ValueError("Tensor images must have shape (3, 32, 32).")
        return tensor

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("image must be a path, PIL image, or torch.Tensor")

    return get_transform()(image)


def predict_scores(image: ImageInput, model: SmallCNN) -> torch.Tensor:
    """Return the model's 10 raw CIFAR-10 output scores (logits) for one image."""
    device = next(model.parameters()).device
    image_tensor = _prepare_image(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        scores = model(image_tensor)

    return scores.squeeze(0).cpu()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument(
        "--weights",
        default="weights/cifar10_small_cnn.pth",
        help="Path to the saved state_dict",
    )
    args = parser.parse_args()

    model = load_model(args.weights)
    scores = predict_scores(args.image, model)
    predicted_index = int(scores.argmax().item())

    print("Scores:", scores)
    print("Predicted class:", CIFAR10_CLASSES[predicted_index])
