import torch
import torch.nn as nn
import torch.nn.functional as F

class Function:
  def __init__(self,params):
    self.params = params
    return

  def __call__(self,x):
    return
  def SetDtype(self,dtype):
    for i in range(len(self.params)):
      self.params[i] = self.params[i].to(dtype)
    return
  def SetDevice(self,device):
    for i in range(len(self.params)):
      self.params[i] = self.params[i].to(device)
    return

class QuadraticFunction(Function):
  def __call__(self,x):
    Q = self.params[0]
    b = self.params[1]
    return 1/2*(Q@x)@x+b@x

class CNN_func(Function):
    def __init__(self, params):
      super().__init__(params)
      self.criterion = nn.CrossEntropyLoss()

    def __call__(self,x):
      params = 0
      weight1 = x[params:params + 16*1*5*5].reshape(16,1,5,5)
      params += 16*1*5*5
      bias1 = x[params:params+16]
      params += 16
      weight2 = x[params:params + 32*16*5*5].reshape(32,16,5,5)
      params += 32*16*5*5
      bias2 = x[params:params + 32]
      params += 32
      W = x[params:params + 8 * 8 * 32 *10].reshape(10,8 * 8 * 32)
      params += 8 * 8 * 32 *10
      b = x[params:]
      params += 10

      z = F.conv2d(input = self.params[0],weight = weight1, bias = bias1,padding = 2)
      z = torch.sigmoid(z)
      z = F.avg_pool2d(input = z , kernel_size= 2)
      z = F.conv2d(input = z, weight = weight2, bias = bias2,padding =2)
      z = torch.sigmoid(z)
      z = F.avg_pool2d(input = z,kernel_size = 2)
      z = z.view(z.size(0), -1)
      z = F.linear(z, W, bias=b)
      return self.criterion(z, self.params[1].to(torch.int64))

    def predict(self,x):
      params = 0
      weight1 = x[params:params + 16*1*5*5].reshape(16,1,5,5)
      params += 16*1*5*5
      bias1 = x[params:params+16]
      params += 16
      weight2 = x[params:params + 32*16*5*5].reshape(32,16,5,5)
      params += 32*16*5*5
      bias2 = x[params:params + 32]
      params += 32
      W = x[params:params + 8 * 8 * 32 *10].reshape(10,8 * 8 * 32)
      params += 8 * 8 * 32 *10
      b = x[params:]
      params += 10

      z = F.conv2d(input = self.params[0],weight = weight1, bias = bias1,padding = 2)
      z = torch.sigmoid(z)
      z = F.avg_pool2d(input = z , kernel_size= 2)
      z = F.conv2d(input = z, weight = weight2, bias = bias2,padding =2)
      z = torch.sigmoid(z)
      z = F.avg_pool2d(input = z,kernel_size = 2)
      z = z.view(z.size(0), -1)
      z = F.linear(z, W, bias=b)
      return torch.argmax(z,dim = 1)

