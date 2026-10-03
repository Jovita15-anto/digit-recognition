"""Image preprocessing for digit recognition (NumPy + OpenCV only, no PyTorch needed).

The recognition steps follow the original project: grayscale -> resize to 8x8 ->
ink brighter than paper (the "255 - image" inversion) -> feed to the CNN.
This module adds what is needed to read *any* digits: automatic ink/paper detection
and splitting an image into one crop per digit.
"""
import cv2
import numpy as np

TILE = 8            # the CNN expects 8 x 8 images
FILL = 1.0          # like the training images, the digit's longer side spans the whole 8 px frame


def to_ink(image):
    """Grayscale image where ink is bright and paper is dark (0), values 0-255.

    Same idea as `255 - gray` in the original code, but it also works for white-on-dark
    images and removes the paper's grey level so the background is a clean 0.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    ink = (255 - gray) if np.median(border) > 127 else gray.copy()
    ink = ink.astype(np.float32)
    edge = np.concatenate([ink[0, :], ink[-1, :], ink[:, 0], ink[:, -1]])
    ink = np.clip(ink - np.median(edge), 0, None)
    if ink.max() > 0:
        ink *= 255 / ink.max()
    return ink.astype(np.uint8)


def find_digit_boxes(ink):
    """Return bounding boxes (x, y, w, h) of the digits in an ink image, left to right."""
    if ink.max() < 40:
        return []
    _, bw = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted(cv2.boundingRect(c) for c in contours)

    # merge pieces of the same digit (e.g. a detached stroke) that overlap horizontally
    merged = []
    for x, y, w, h in boxes:
        if merged:
            mx, my, mw, mh = merged[-1]
            overlap = min(mx + mw, x + w) - max(mx, x)
            if overlap > 0.3 * min(mw, w):
                nx, ny = min(mx, x), min(my, y)
                merged[-1] = (nx, ny, max(mx + mw, x + w) - nx, max(my + mh, y + h) - ny)
                continue
        merged.append((x, y, w, h))

    if not merged:
        return []
    tallest = max(h for _, _, _, h in merged)
    kept = [b for b in merged if b[3] >= 0.3 * tallest and b[2] * b[3] >= 12]
    return sorted(kept)


def box_to_tile(ink, box):
    """Crop one digit, centre it in a square with a margin, and shrink to 8 x 8."""
    x, y, w, h = box
    side = max(int(round(max(w, h) / FILL)), 2)
    pad = side
    padded = cv2.copyMakeBorder(ink, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    cx, cy = x + w // 2 + pad, y + h // 2 + pad
    x0, y0 = cx - side // 2, cy - side // 2
    square = padded[y0:y0 + side, x0:x0 + side]
    tile = cv2.resize(square, (TILE, TILE), interpolation=cv2.INTER_AREA)
    if tile.max() > 0:
        tile = (tile.astype(np.float32) * 255 / tile.max()).astype(np.uint8)
    return tile


def split_digits(image):
    """Image -> list of (box, 8x8 uint8 tile), one per digit, left to right."""
    ink = to_ink(image)
    return [(b, box_to_tile(ink, b)) for b in find_digit_boxes(ink)]
