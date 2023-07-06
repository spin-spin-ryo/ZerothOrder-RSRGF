import torch
import os
from environments import DATAPATH
from utils import *
from function import *


def generate(mode,properties):
    if mode == "test":
        f ,x0 = generate_test(properties)
    elif mode == "Quadratic":
        f,x0 = generate_quadratic(properties)
    
    else:
        raise ValueError("No functions.")

    return f,x0

def generate_test(properties):
    dim = int(properties["dim"])
    Q = torch.eye(dim)
    b = torch.zeros(dim)
    x0 = torch.ones(dim)
    params = [Q,b]
    f = QuadraticFunction(params=params)
    return f,x0

def generate_quadratic(properties):
    property = properties["property"] 
    dim = int(properties["dim"])
    rank = int(properties["rank"])
    savepath = os.path.join(DATAPATH,"quadratic",property)
    filename_Q = f"Q_{dim}_{rank}.pth"
    filename_b = f"b_{dim}_{rank}.pth"
    filename_x0 = f"x0_{dim}_{rank}.pth"
    if os.path.exists(os.path.join(savepath,filename_Q)):
        Q = torch.load(os.path.join(savepath,filename_Q))
        b = torch.load(os.path.join(savepath,filename_b))
        x0 = torch.load(os.path.join(savepath,filename_x0))
        
    else:
        os.makedirs(savepath,exist_ok=True)
        if property == "convex":
            Q = generate_semidefinite(dim,rank)
            b = torch.randn(dim)
            x0 = 10*torch.randn(dim)
        elif property == "sconvex":
            Q = generate_definite(dim)
            b = torch.randn(dim)
            x0 = 10*torch.randn(dim)
        elif property == "nonconvex":
            Q = generate_symmetric(dim)
            b = torch.randn(dim)  
            x0 = 10*torch.randn(dim)  
        else:
            raise ValueError("There is no property.")
        torch.save(Q,os.path.join(savepath,filename_Q))
        torch.save(b,os.path.join(savepath,filename_b))
        torch.save(x0,os.path.join(savepath,filename_x0))
    params = [Q,b]
    f = QuadraticFunction(params=params)
    return f,x0