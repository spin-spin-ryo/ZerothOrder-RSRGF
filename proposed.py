import torch
import os
from torch.autograd.functional import hessian
from torch.utils.data import TensorDataset, DataLoader
from function import CNN_func
import numpy as np
import matplotlib.pyplot as plt
import time

if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

print(device)

DATAPATH = "./data"
data = torch.load(os.path.join(DATAPATH,"CNN","images.pth"))
label = torch.load(os.path.join(DATAPATH,"CNN","labels.pth"))

dtype = torch.float64
torch.cuda.empty_cache()

dataset = TensorDataset(data,label)
batch_size = 4000
dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = True)

class subspace_f:
  def __init__(self,func):
    self.x = None
    self.P = None
    self.func = func
  
  def set(self,x,P):
    self.x = x
    self.P = P
  
  def __call__(self,d):
    return self.func.value(self.x + self.P@d)
  
def GetMinimumEig(H):
  eigenvalues = torch.linalg.eigvalsh(H)
  minimum_eigenvalue = eigenvalues[0].item()
  return minimum_eigenvalue

def compute_min_eigen(matrix):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

    # Find the index of the minimum eigenvalue
    min_eigenvalue_index = torch.argmin(eigenvalues)
    min_eigenvalue = eigenvalues[min_eigenvalue_index].item()

    # Retrieve the corresponding eigenvector
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]

    return min_eigenvalue, min_eigenvector

  
dim = 33738
reduced_dim = 10
params_f = [data[:40000],label[:40000]]
f = CNN_func(params=params_f)
obj = subspace_f(f)
x = torch.load(os.path.join(DATAPATH,"CNN","init_param.pth")).to(device)
c1 = 1.5
c2 = 0.1
t = 1
alpha = 0.3
beta = 0.8
epoch_num = 2000
losses = []
total_time = 0
init_lr = 10
for epoch in range(epoch_num):
  P = torch.randn(dim,reduced_dim)/(dim**(0.5))
  P = P.to(device)
  total_loss = 0
  start_epoch = time.time()
  y = x.clone().detach().to(device)
  y.requires_grad_(True)
  loss = f.value(y)
  loss.backward()

   
  print(f"Norm:{torch.linalg.norm(y.grad)}")
  if torch.linalg.norm(y.grad) > 1e-1:
    dk = - y.grad
    print("GRADIENT")
    flag = True
  else:
      d = torch.zeros(reduced_dim).to(device)
      d.requires_grad_(True)
      obj.set(x,P)
      hess = hessian(obj,d)
      if total_hess is None:
        total_hess = hess
      else:
        total_hess += hess
    total_hess /= len(dataloader)
    flag = False
    min_eig,min_vec = compute_min_eigen(total_hess)
    if min_eig <= 0:
      dk = -torch.sign(y.grad@P@min_vec)*P@min_vec
      print("NC")
    else:
      dk = -P@torch.linalg.inv(total_hess)@P.transpose(0,1)@y.grad
      print("NEWTON")
  with torch.no_grad():
      if flag:
        lr = init_lr
      else:
        lr = 10
      f.params = [data[:40000].to(device),label[:40000].to(device)]
      f1 = f.value(x)
      while f1 - f.value(x + lr*dk) < -alpha*lr*y.grad@dk:
          lr *= beta  
      x += lr*dk
      if flag:
        if lr == init_lr:
          init_lr *=2
        else:
          init_lr = lr
  losses.append(f1.cpu())
  print(losses[-1])
  print("")

plt.plot(np.arange(len(losses)),losses)
plt.savefig("./results/proposed_plot.png")