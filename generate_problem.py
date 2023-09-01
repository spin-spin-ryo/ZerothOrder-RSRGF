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
    elif mode == "max-linear":
        f,x0 = generate_max_linear(properties)
    elif mode == "piecewise-linear":
        dim = int(properties["dim"])
        f = piecewise_linear()
        x0 = torch.zeros(dim)
    elif mode == "subspace-norm":
        f,x0 = generate_subspace(properties)
    elif mode == "subspace-norm local":
        f,x0 = generate_subspace(properties,local = True)
    elif mode == "regularized test":
        f ,x0 = generate_test(properties)
        f = generate_regularized(f,properties)
    elif mode == "regularized test local":
        f ,x0 = generate_test(properties,local = True)
        f = generate_regularized(f,properties)
    elif mode == "regularized Quadratic":
        f,x0 = generate_quadratic(properties)
        f = generate_regularized(f,properties)
    elif mode == "regularized Logistic":
        f,x0 = generate_logistic(properties)
        f = generate_regularized(f,properties)
    else:
        raise ValueError("No functions.")
    return f,x0

def generate_max_linear(properties):
    dim = int(properties["dim"])
    number = int(properties["number"])
    savepath = os.path.join(DATAPATH,"max-linear")
    filename_A = f"A_{dim}_{number}.pth"
    filename_b = f"b_{dim}_{number}.pth"
    filename_x0 = f"x0_{dim}_{number}.pth"
    if os.path.exists(os.path.join(savepath,filename_A)):
        A = torch.load(os.path.join(savepath,filename_A))
        b = torch.load(os.path.join(savepath,filename_b))
        x0 = torch.load(os.path.join(savepath,filename_x0))
    else:
        os.makedirs(savepath,exist_ok=True)
        A = torch.randn(number,dim)
        b = torch.ones(number)
        x0 = torch.randn(dim)*10
        torch.save(A,os.path.join(savepath,filename_A))
        torch.save(b,os.path.join(savepath,filename_b))
        torch.save(x0,os.path.join(savepath,filename_x0))
    params = [A,b]
    f = max_linear(params)
    return f,x0


    

def generate_test(properties,local = False):
    dim = int(properties["dim"])
    if not local:
        x0 = torch.ones(dim)
    else:
        x0 = torch.ones(dim) / dim
    f = test_function()
    return f,x0

def generate_subspace(properties,local = False):
    dim = int(properties["dim"])
    subspace_dim = int(properties["subspace"])
    ord = int(properties["ord"])
    f = subspace_norm([torch.tensor(subspace_dim),torch.tensor(ord)])
    if not local:
        x0 = torch.ones(dim)
    else:
        x0 = torch.ones(dim)/dim
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

def generate_logistic(properties):
    dim = properties["dim"]
    data_num = properties["data_num"]
    savepath = os.path.join(DATAPATH,"logistic")
    filename_A = f"A_{dim}_{data_num}.pth"
    filename_b = f"b_{dim}_{data_num}.pth"
    filename_x0 = f"x0_{dim}_{data_num}.pth"
    if os.path.exists(os.path.join(savepath,filename_A)):
        A = torch.load(os.path.join(savepath,filename_A))
        b = torch.load(os.path.join(savepath,filename_b))
        x0 = torch.load(os.path.join(savepath,filename_x0))
    else:
        os.makedirs(savepath,exist_ok=True)
        A = torch.randn(data_num,dim)
        b = generate_zeroone(data_num)
        x0 = torch.randn(dim)*10
        torch.save(A,os.path.join(savepath,filename_A))
        torch.save(b,os.path.join(savepath,filename_b))
        torch.save(x0,os.path.join(savepath,filename_x0))
    params = [A,b]
    f = logistic(params)
    return f,x0

def generate_regularized(f,properties):
    p = torch.tensor(properties["ord"]).to(torch.int32)
    coef = torch.tensor(properties["coef"])
    fused_flag = properties["fused"]
    
    if fused_flag:
        dim = properties["dim"]
        A = generate_fusedmatrix(dim)
    else:
        A = None
    params = [p,coef,A]
    f_r = regularizedfunction(f,params)
    return f_r
    