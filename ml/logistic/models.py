import torch
import torch.nn.functional as F
import torch.nn as nn


class Logistic(nn.Module):
    def __init__(self, features_num,class_num) -> None:
        super().__init__()
        self.path = nn.Sequential(nn.Linear(in_features=features_num,out_features=class_num,bias=True))
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        return self.path(x)
    
    def pred(self,x,y):
        x = self.forward(x)
        return self.softmax(x)
