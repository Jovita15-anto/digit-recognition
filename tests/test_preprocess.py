import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import split_digits, to_ink  # noqa: E402


def typed(text, dark_on_light=True):
    img = np.full((120, 60 * len(text) + 40, 3), 255, np.uint8)
    cv2.putText(img, text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (30, 30, 30), 6, cv2.LINE_AA)
    return img if dark_on_light else 255 - img


def test_finds_one_box_per_digit():
    assert len(split_digits(typed("2025"))) == 4
    assert len(split_digits(typed("7"))) == 1


def test_works_on_light_digits_on_dark_background():
    assert len(split_digits(typed("2025", dark_on_light=False))) == 4


def test_tiles_are_8x8_with_bright_ink():
    for _, tile in split_digits(typed("30")):
        assert tile.shape == (8, 8) and tile.max() == 255


def test_ink_polarity_is_normalised():
    a, b = to_ink(typed("5")), to_ink(typed("5", dark_on_light=False))
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 5


def test_blank_image_has_no_digits():
    assert split_digits(np.full((100, 100, 3), 255, np.uint8)) == []
