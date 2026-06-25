import torch
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import os
import set_model as model_factory
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

#if __name__ == '__main__':
 #   main()

def main():
    pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["simple", "vgg"],
    required=True,
    help="Model architecture to use"
)

parser.add_argument(
    "--image",
    required=False,
    help="Path to input image"
)

args = parser.parse_args()

MODEL_NAME = args.model
MODEL_PATH = os.path.join(BASE_DIR, f"mnist_{MODEL_NAME}.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model_factory.get_model(device=device, model_name = MODEL_NAME, train_mode=False, weights_path=MODEL_PATH)

print(f"Model loaded: {MODEL_PATH}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

if args.image:
    file_path = args.image
else:
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
        messagebox.showinfo("Info", "File wasn't chosen")
        exit()

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
tensor = tensor.unsqueeze(0)
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