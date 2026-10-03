import cv2
import torch
import torch.nn as nn

# Create the same model architecture used during training
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Flatten(),

    nn.Linear(256, 128),
    nn.ReLU(),

    nn.Linear(128, 10)
)

# Load trained weights
model.load_state_dict(
    torch.load("model_weights_10class.pth", weights_only=True)
)

# Read image
image = cv2.imread("two.jpeg")

if image is None:
    print("Error: two.jpeg could not be loaded.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize to 8 × 8
resized = cv2.resize(gray, (8, 8))

# Add channel dimension
imag_f = resized.reshape(1, 8, 8)

# Invert image
req_image = 255 - imag_f

# Add batch and channel dimensions
row_image = req_image.reshape(1, 1, 8, 8)

# Convert to PyTorch tensor
tor_image = torch.FloatTensor(row_image)

# Prediction
model.eval()

with torch.no_grad():
    Yp = model(tor_image)

    print("Output shape:", Yp.shape)

    sx = torch.softmax(Yp, dim=1)

    predicted_img = torch.argmax(sx, dim=1)

    print("Predicted class:", predicted_img.item())