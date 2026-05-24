import torch
from torchvision import datasets
import os

os.makedirs('mnist_test_samples', exist_ok=True)


test_dataset = datasets.MNIST(root='./data', train=False, download=False)

print(f"total images: {len(test_dataset)}")

for i in range(10):
    image, label = test_dataset[i]
    file_name = f'mnist_test_samples/test_img_{i}_label_{label}.png'
    image.save(file_name)

print("saved in 'mnist_test_samples'")