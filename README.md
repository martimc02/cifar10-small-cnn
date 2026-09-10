# CIFAR-10 CNN

Small convolutional neural network trained on CIFAR-10 using PyTorch.

The code is based on the PyTorch CIFAR-10 tutorial. The network has two convolutional layers followed by three fully connected layers and gives 10 output scores, one for each CIFAR-10 class.

## Files

- `model.py` - CNN architecture
- `train.py` - training and saving the weights
- `evaluate.py` - test accuracy
- `inference.py` - function that receives an image and returns the output scores

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

The CIFAR-10 dataset is downloaded automatically and the weights are saved in:

```text
weights/cifar10_small_cnn.pth
```

## Test

```bash
python evaluate.py
```

## Output scores for one image

```python
from inference import load_model, predict_scores

model, device = load_model()
scores = predict_scores("image.png", model, device)
print(scores)
```

The returned tensor contains the 10 scores produced by the last layer of the network.
