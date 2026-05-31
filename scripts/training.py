import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import set_model as model_factory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'mnist_model.pth')

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_factory.get_model(train_mode=True, weights_path=None, device=device)
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

try:
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(f"Succsess saving {len(train_dataset)} pictures.")
    print(f"Use the path: {DATA_DIR}")
except Exception as e:
    print(f"No data in {DATA_DIR}")
    print(f"Struct check: data/MNIST/raw/")
    print(f"Origin mistake: {e}")
    exit()

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
criterion = nn.CrossEntropyLoss()

history = {'loss': [], 'acc': [], 'lr': []}

print("\nstart...")
for epoch in range(1, 6):
    running_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    
    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)
    history['lr'].append(current_lr)
    
    print(f'Epoch {epoch}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, LR: {current_lr}')
    scheduler.step()

epochs_range = range(1, len(history['loss']) + 1)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, history['loss'], 'r-o', label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epochs_range, history['acc'], 'g-s', label='Accuracy')
plt.title('Training Accuracy (%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(epochs_range, history['lr'], 'b-^', label='LR')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('LR Value')
plt.yscale('log') 
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, 'training_stats.png')
plt.savefig(save_path)
print(f"Graph saved in: {save_path}")
plt.show() 
plt.close()

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nsave")
print(f"Grap save in: {os.path.join(RESULTS_DIR, 'training_stats.png')}")
print(f"Model represent by: {MODEL_SAVE_PATH}")