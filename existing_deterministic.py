import torch
import os
from torch.autograd.functional import hessian
from torch.utils.data import TensorDataset, DataLoader
from function import CNN_func
import numpy as np
import matplotlib.pyplot as plt

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

dataset = TensorDataset(data[:40000],label[:40000])
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


dim = 33738
reduced_dim = 100
params_f = [data[:40000].to(device),label[:40000].to(device)]
f = CNN_func(params=params_f)
obj = subspace_f(f)

x = torch.load(os.path.join(DATAPATH,"CNN","init_param.pth"))
x.requires_grad_(True)
lr = 10
epoch_num = 2001
losses = []
grads = []
eigs = []
for epoch in range(epoch_num):
    P =torch.randn(reduced_dim,dim)/(dim**0.5)
    P = P.to(device)
    x.grad = None
    loss = f.value(x)
    loss.backward()
    with torch.no_grad():
        x -= lr*P.transpose(0,1)@P@x.grad
    losses.append(loss.item())
    grads.append(torch.linalg.norm(x.grad).cpu())

    if epoch%20 == 0:
      P =torch.randn(10,dim)/(dim**0.5)
      P = P.to(device)
      d = torch.zeros(10).to(device)
      obj.set(x.clone().detach(),P.transpose(0,1))
      total_hess = None
      for batch in dataloader:
        images = batch[0]
        labels =  batch[1]
        obj.func.params = [images.to(device),labels.to(device)]
        if total_hess is None:
          total_hess = hessian(obj,d)
        else:
          total_hess += hessian(obj,d)
      min_lambda = GetMinimumEig(total_hess/len(dataloader))
      eigs.append(min_lambda)
      f.params = [data[:40000].to(device),label[:40000].to(device)]
      plt.plot(np.arange(len(losses)),losses)
      plt.savefig("./results/test_plot_existing2.png")
      plt.close()
      plt.plot(np.arange(len(grads)),grads)
      plt.savefig("./results/grad_plot_existing2.png")
      plt.close()

      plt.plot(np.arange(len(eigs)),eigs)
      plt.savefig("./results/eig_plot_existing2.png")
      plt.close()
        

   
