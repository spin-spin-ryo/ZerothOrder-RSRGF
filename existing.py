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
batch_size = 1024
dataloader = DataLoader(dataset,batch_size = batch_size,shuffle = True)

  
dim = 33738
params_f = [data[:10],label[:10]]
f = CNN_func(params=params_f)
x = torch.load(os.path.join(DATAPATH,"CNN","init_param.pth"))
x.requires_grad_(True)
lr = 0.5
epoch_num = 2000
losses = []
for epoch in range(epoch_num):
    for batch in dataloader:
        images = batch[0]
        labels = batch[1]
        images = images.to(device)
        labels = labels.to(device)
        f.params = [images,labels]
        x.grad = None
        loss = f.value(x)
        loss.backward()
        with torch.no_grad():
            x -= lr*x.grad
    
    if epoch%20== 0:
      f.params = [data[40000:].to(device),label[40000:].to(device)]
      with torch.no_grad():
        z = f.predict(x)
        print(torch.sum(z==f.params[1])/len(z))


plt.plot(np.arange(len(losses)),losses)
plt.savefig("./results/test_plot_existing.png")