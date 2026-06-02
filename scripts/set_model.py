import torch
import torch.nn as nn
import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

class SimpleNet(nn.Module): 
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) 
        
    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(self.relu(self.fc1(x)))
    
    #def set_mode =(self, train_mode: bool):
     #   if train_mode:
      #      self.train()
       # else:
        #    self.eval() 

class MiniVGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
def get_model(device: torch.device, model_name: str, train_mode: bool = True, weights_path: str | None = None) -> SimpleNet:

    if model_name == "simple":
        model = SimpleNet()

    elif model_name == "vgg":
        model = MiniVGG()

    model = model.to(device)

    if weights_path is not None:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"{weights_path} - file not found")
        weights = torch.load(weights_path, map_location=device)
        model.load_state_dict(weights)

    if train_mode:
        model.train()
    else:
        model.eval()
    return model 