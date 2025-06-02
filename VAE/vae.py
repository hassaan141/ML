import torch
from torch import nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self,x):
        pass
    
    def decode(self,z):

    def forward(self,x):


if __name__=="__main__":
    x = torch.randn(1,784)
    