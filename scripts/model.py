import torch
import torch.nn as nn
import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

#def weights_file_exist(root_weight str, root_dir: str = DIR_PATH) -> str:
 #   for 
class SimpleNet(nn.Module): 
    def __init__(self):
        super(SimpleNet, self).__init__()
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
        
def get_model(train_mode: bool = True, weights_path: str | None = None) -> SimpleNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)

    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"{weights_path} - not found")
        weights = torch.load(weights_path, map_location=device)
        model.load_state_dict(weights)

    if train_mode:
        model.train()
    else:
        model.eval()
    return model 