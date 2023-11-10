from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from solver import BackTrackingPGD,projection_ball2,BackTrackingAccerelatedPGD
from optim_method import logger

class Function:
  def __init__(self,params = []):
    self.params = params
    return

  def __call__(self,x):
    return
  
  def SetDtype(self,dtype):
    for i in range(len(self.params)):
      try:
        self.params[i] = self.params[i].to(dtype)
      except:
        print(type(self.params[i]))
    return
  
  def SetDevice(self,device):
    for i in range(len(self.params)):
      try:
        self.params[i] = self.params[i].to(device)
      except:
        print(type(self.params[i]))
    return

class QuadraticFunction(Function):
  def __call__(self,x):
    Q = self.params[0]
    b = self.params[1]
    return 1/2*(Q@x)@x+b@x

class test_function(Function):
  def __call__(self,x):
    return 1/2*x@x

class max_linear(Function):
  def __call__(self, x):
    A = self.params[0]
    b = self.params[1]
    return torch.max(A@x + b)

class piecewise_linear(Function):
  def __call__(self,x):
    return torch.abs(1-x[0]) + torch.sum(torch.abs( 1 + x[1:] - 2*x[:-1]))
  
class norm_function(Function):
  def __call__(self, x):
    Q = self.params[0]
    b = self.params[1]
    p = self.params[2]
    return torch.linalg.norm(Q@x - b,ord = p)

class logistic(Function):
  def __call__(self,x):
    # Xの最後には列には1だけのものがある
    # yは-1,1で
    X = self.params[0]
    y = self.params[1]
    a = X@x
    return torch.mean(torch.log(1 + torch.exp(-y*a)))

class robust_logistic(Function):
  def __init__(self, params=[],delta = 0.1,inner_iteration = 100000,subproblem_eps = 1e-5):
    self.inner_iteration = inner_iteration
    self.subproblem_eps = subproblem_eps
    self.projection = False
    self.delta = delta

    super().__init__(params)

  def __call__(self, w,u = None):
    # Xの最後には列には1だけのものがある
    # yは-1,1で
    # noiseは1以外のものに対して
    
    func,prox,x0 = self.__set__inner_call__(w=w,u=u)
    
    return -self.solve_subproblem(func=func,prox=prox,x0=x0,eps=self.subproblem_eps,iteration=self.inner_iteration)
   
  def __set__inner_call__(self,w,u=None):
    X = self.params[0]
    data_num,feature_num = X.shape
    self._a_ = X@w
      
    if self.projection:
      # u はmu_k/sqrt{n} u
      def func(x_input):
        return -self.__inner_projection_call__(x_input[0],x_input[1:],u=u)
      def prox(x,t):
        return projection_ball2(x,t,r = self.delta)
      x0 = torch.zeros(u.shape[0]+1,device=X.device,dtype = X.dtype)

    else:
      def func(x_input):
        return -self.__inner_call__(x_input,w=w)
      
      def prox(x,t):
        return projection_ball2(x,t,r = self.delta)
      x0 = torch.zeros(feature_num,device=X.device,dtype = X.dtype)
    
    return func,prox,x0
    
  def __inner_call__(self,delta_X,w):
    y = self.params[1]
    a = self._a_ + delta_X@w
    return torch.mean(torch.log(1 + torch.exp(-y*a)))
  
  def __inner_projection_call__(self,alpha,beta,u):
    y = self.params[1]
    a = self._a_ + alpha + u@beta
    return torch.mean(torch.log(1 + torch.exp(-y*a)))
  
  def solve_subproblem(self,func,prox,x0,eps=1e-6,iteration = 10000):
    solver = BackTrackingAccerelatedPGD(func=func,prox=prox)
    solver.__iter__(x0=x0,iteration=iteration,eps=eps,restart=True)
    return solver.get_function_value()
  
  def solve_subproblem_solution(self,func,prox,x0,eps=1e-6,iteration = 10000):
    solver = BackTrackingAccerelatedPGD(func=func,prox=prox)
    solver.__iter__(x0=x0,iteration=iteration,eps=eps,restart=True)
    return solver.get_solution()
  

  def get_subproblem_solution(self, w,u = None):

    func,prox,x0 = self.__set__inner_call__(w=w,u=u)
    
    return self.solve_subproblem_solution(func=func,prox=prox,x0=x0,eps=self.subproblem_eps,iteration=self.inner_iteration)
   
class softmax(Function):
  def __call__(self,x,eps = 1e-12):
    X = self.params[0]
    y = self.params[1]
    data_num,feature_num = X.shape
    _,class_num = y.shape
    W = x[:feature_num*class_num].reshape(feature_num,class_num)
    Z = X@W
    sum_Z = torch.logsumexp(Z,1)
    sum_Z = sum_Z.unsqueeze(1)
    out1 = -Z + eps + sum_Z
    return torch.mean(torch.sum(out1*y,dim = 1))

class subspace_norm(Function):
  def __call__(self,x):
    r = self.params[0]
    p = self.params[1]
    return torch.linalg.norm(x[:r],ord = p)
  
  def SetDtype(self,dtype):
    for i in range(len(self.params)):
      if self.params[i] is not None:
        self.params[i] = self.params[i].to(torch.int64)
    return  

class LinearRegression(Function):
  def __init__(self, params=[],bias = False):
    super().__init__(params)
    self.bias = bias

  def __call__(self, x):
    A = self.params[0]
    b = self.params[1]
    if not self.bias:
      return torch.linalg.norm(A@x - b)**2
    else:
      return torch.linalg.norm(A@x[:-1] + x[-1] - b)**2

class NMF(Function):
  def __call__(self, x):
    W = self.params[0]
    height,width = W.shape
    rank = self.params[1]
    U = x[:height*rank].reshape(height,rank)
    V = x[height*rank:].reshape(rank,width)
    return torch.linalg.norm(U@V - W)**2

  
  def SetDtype(self, dtype):
    super().SetDtype(dtype)
    self.params[1] = self.params[1].to(torch.int32)
    return 

class rosenbrock(Function):
  def __call__(self,x):
    super().__call__(x)

class adversarial(Function):
  def __init__(self, params=[]):
    from ml.logistic.models import Logistic
    super().__init__(params)
    X = self.params[0]
    y = self.params[1]
    epoch_num = self.params[2]
    data_num = X.shape[0]
    features_num = X.shape[1]
    class_num = torch.max(y)+1
    self.model = Logistic(features_num=features_num,class_num=class_num)
    model_path = f"./ml/logistic/checkpoints/model_epoch{epoch_num}.pth"
    self.model.load_state_dict(torch.load(model_path))
  
  def __call__(self, x):
    # x : (features_num)
    X = self.params[0]
    y = self.params[1]
    coef = self.params[3]
    threshold = 0
    y_onehot = F.one_hot(y,num_classes = self.class_num).to(torch.bool)
    not_y_onehhot = torch.logical_not(y_onehot)
    with torch.no_grad():
      scores = self.model.loss(X+x)
      true_scores = scores[y_onehot]
      max_not_true_scores = torch.max(scores[not_y_onehhot].reshape(self.data_num,self.class_num-1),dim = 1).values
      untarget_loss = true_scores - max_not_true_scores
      untarget_loss[untarget_loss<0]=0
      return untarget_loss + coef*(x@x)

  def SetDevice(self, device):
    self.model = self.model.to(device)
    return super().SetDevice(device)

  def SetDtype(self, dtype):
    self.model = self.model.to(dtype)
    super().SetDtype(dtype)
    self.params[1] = self.params[1].to(torch.int64)
    return

class robust_adversarial(Function):
  def __init__(self, params=[],delta=0.1,subproblem_eps = 1e-6,inner_iteration =10000):
    # params = [X,y]
    super().__init__(params)
    self.subproblem_eps = subproblem_eps
    self.inner_iteration = inner_iteration
    self.delta = delta
    

  def __call__(self,x,eps = 1e-12):
    X = self.params[0]
    y = self.params[1]
    data_num,feature_num = X.shape
    _,class_num = y.shape
    
    
    def func(x_input):
      return -self.inner_func(x_input,x=x)
    
    def prox(x,t):
      return projection_ball2(x,t,r = self.delta)
    
    x0 = torch.zeros(feature_num,device=X.device,dtype = X.dtype)
    delta_X = self.solve_subproblem(func=func,prox=prox,x0=x0,eps=self.subproblem_eps,iteration=self.inner_iteration)
    W = x[:feature_num*class_num].reshape(feature_num,class_num)
    Z_ = X@W
    delta_Z = delta_X@W
    Z = Z_ + delta_Z + x[feature_num*class_num:]
    sum_Z = torch.logsumexp(Z,1)
    sum_Z = sum_Z.unsqueeze(1)
    out1 = -Z + eps + sum_Z
    return torch.mean(torch.sum(out1*y,dim = 1))

  def inner_func(self,delta_X,x,eps = 1e-12):
    X = self.params[0]
    y = self.params[1]
    data_num,feature_num = X.shape
    _,class_num = y.shape
    W = x[:feature_num*class_num].reshape(feature_num,class_num)
    Z_ = X@W
    # [datanum,classnum]
    delta_Z = delta_X@W
    # [classnum]
    Z = Z_ + delta_Z + x[feature_num*class_num:]
    sum_Z = torch.logsumexp(Z,1)
    sum_Z = sum_Z.unsqueeze(1)
    out1 = -Z + eps + sum_Z
    return torch.mean(torch.sum(out1*y,dim = 1))

    
  
  def solve_subproblem(self,func,prox,x0,eps=1e-6,iteration = 10000):
    solver = BackTrackingAccerelatedPGD(func=func,prox=prox)
    solver.__iter__(x0=x0,iteration=iteration,eps=eps,restart=True)
    return solver.get_solution()

class regularizedfunction(Function):
  def __init__(self,f,params):
    self.f = f
    assert len(params) ==3
    self.params = params
    return

  def __call__(self,x):
    p = self.params[-3]
    l = self.params[-2]
    A = self.params[-1]
    if A is not None:
      return self.f(x) + l*torch.linalg.norm(A(x),ord = p)
    else:
      return self.f(x) + l*torch.linalg.norm(x,ord = p)
  
  def SetDevice(self, device):
    self.f.SetDevice(device)
    return super().SetDevice(device)

  def SetDtype(self, dtype):
    self.f.SetDtype(dtype)
    return super().SetDtype(dtype)

class projectionregularizedfunction(regularizedfunction):
  def __call__(self, x,u=None):
    p = self.params[-3]
    l = self.params[-2]
    A = self.params[-1]
    if A is not None:
      return self.f(x,u) + l*torch.linalg.norm(A(x),ord = p)
    else:
      return self.f(x,u) + l*torch.linalg.norm(x,ord = p)
  
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

