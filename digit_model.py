"""CNN digit classifier (the architecture and weights from the original project)."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from preprocess import split_digits

WEIGHTS = Path(__file__).parent / "model_weights_10class.pth"


def build_model():
    """Same architecture used during training."""
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten(),

        nn.Linear(256, 128),
        nn.ReLU(),

        nn.Linear(128, 10),
    )


@lru_cache(maxsize=1)
def load_model():
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS, weights_only=True))
    model.eval()
    return model


def classify_tile(tile):
    """8x8 image (ink bright, values 0-255) -> array of 10 class probabilities."""
    tensor = torch.FloatTensor(np.asarray(tile, dtype=np.float32).reshape(1, 1, 8, 8))
    with torch.no_grad():
        scores = load_model()(tensor)
        probabilities = torch.softmax(scores, dim=1)
    return probabilities[0].numpy()


def _result(box, tile):
    probs = classify_tile(tile)
    digit = int(probs.argmax())
    return {"digit": digit, "confidence": float(probs[digit]), "probs": probs, "tile": tile, "box": box}


def recognize(image):
    """Find every digit in a BGR image and classify each one with the CNN."""
    digits = [_result(box, tile) for box, tile in split_digits(image)]
    return {"number": "".join(str(d["digit"]) for d in digits), "digits": digits}


def classify_whole_image(image):
    """Original method: treat the whole image as ONE digit (grayscale -> 8x8 -> invert)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tile = 255 - cv2.resize(gray, (8, 8))
    h, w = gray.shape
    digits = [_result((0, 0, w, h), tile)]
    return {"number": str(digits[0]["digit"]), "digits": digits}


def annotate(image, digits):
    """Draw a box and the predicted digit on every detected digit; returns an RGB image."""
    out = image.copy()
    t = max(2, round(max(out.shape[:2]) / 250))
    for d in digits:
        x, y, w, h = d["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (232, 71, 53), t)  # BGR -> blue-ish accent in RGB
        cv2.putText(out, str(d["digit"]), (x, max(y - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    t * 0.55, (232, 71, 53), t, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
