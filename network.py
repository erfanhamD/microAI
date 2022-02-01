import torch.nn as nn

class Lift_base_network(nn.Module):
    def __init__(self):
        super(Lift_base_network, self).__init__()
        self.block = nn.Sequential(
        nn.Linear(5, 20),
        nn.Tanh(),
        nn.Linear(20, 8),
        nn.Tanh(),
        nn.Linear(8, 2)
        )
    def forward(self, x):
        x = self.block(x)
        return x