# Small CNN on CIFAR-10

A small PyTorch project that trains a convolutional neural network on CIFAR-10, saves the learned weights, and provides a function that takes one image and returns the model's raw output scores (logits).

## Architecture

The network follows the small CNN used in the official PyTorch CIFAR-10 tutorial:

- `Conv2d(3, 6, 5)` + ReLU + MaxPool
- `Conv2d(6, 16, 5)` + ReLU + MaxPool
- `Linear(400, 120)` + ReLU
- `Linear(120, 84)` + ReLU
- `Linear(84, 10)`

The final layer returns 10 logits, one for each CIFAR-10 class.

## Project structure

```text
cifar10-small-cnn/
├── model.py
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── weights/
│   └── .gitkeep
├── .gitignore
└── README.md
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and save the weights

```bash
python train.py
```

The CIFAR-10 dataset is downloaded automatically. The model trains for 2 epochs using SGD with momentum, following the small-network setup from the PyTorch tutorial.

After training, the learned parameters are saved as a PyTorch `state_dict` at:

```text
weights/cifar10_small_cnn.pth
```

## Evaluate

```bash
python evaluate.py
```

This loads the saved weights and reports accuracy on the CIFAR-10 test set.

## Return output scores for one image

The requested function is `predict_scores` in `inference.py`:

```python
from inference import load_model, predict_scores

model = load_model("weights/cifar10_small_cnn.pth")
scores = predict_scores("my_image.png", model)
print(scores)
```

The function returns a tensor with 10 raw model scores:

```text
tensor([score_0, score_1, ..., score_9])
```

The scores correspond to:

```python
("plane", "car", "bird", "cat", "deer",
 "dog", "frog", "horse", "ship", "truck")
```

The predicted class is the index with the largest score:

```python
predicted_class = scores.argmax().item()
```

You can also run inference from the command line:

```bash
python inference.py path/to/image.png
```

## GPU support

The code automatically uses CUDA when an NVIDIA GPU is available; otherwise it uses the CPU. Model parameters and batches are always moved to the same device.
