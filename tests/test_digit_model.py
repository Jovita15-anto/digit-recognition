import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from digit_model import classify_whole_image, recognize  # noqa: E402


def read(name):
    return cv2.imread(str(ROOT / name))


def test_original_method_on_sample_images():
    assert classify_whole_image(read("zero.jpeg"))["number"] == "0"
    assert classify_whole_image(read("two.jpeg"))["number"] == "2"


def test_digit_detection_on_sample_images():
    assert recognize(read("zero.jpeg"))["number"] == "0"
    assert recognize(read("two.jpeg"))["number"] == "2"


def test_multi_digit_samples():
    assert recognize(read("samples/number_2025.png"))["number"] == "2025"
    assert recognize(read("samples/number_7305.png"))["number"] == "7305"
