import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(BASE_DIR, 'mnist_model.pth')


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(self.relu(self.fc1(x)))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = SimpleNet().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print(f"Model loaded: {MODEL_PATH}")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select the image",
    filetypes=[
        ("Image files", "*.png *.jpg *.jpeg *.bmp"),
        ("All files", "*.*")
    ]
)

if not file_path:
    messagebox.showinfo("Info", "File was`t chosen")
    exit()

image = Image.open(file_path).convert("L")
print(f"Mode: {image.mode}, size:{image.size}")
#plt.imshow(image, cmap='gray')
#image = ImageOps.invert(image)
#plt.imshow(image, cmap='gray')

image = image.resize((28, 28), resample=1)
print(f"Mode: {image.mode}, size:{image.size}")
display_image = image
tensor = transform(image)
tensor = tensor.to(device)


with torch.no_grad():

    output = model(tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][predicted_class].item() * 100


print("\n=== Result ===")
print(f"Predicted digit: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")


plt.imshow(display_image, cmap='gray')
plt.title(
    f"Prediction: {predicted_class} ({confidence:.2f}%)"
)
plt.axis('off')
plt.show()