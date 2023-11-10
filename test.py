from function import robust_logistic
import torch
import time
import torch.nn.functional as F

device = "cpu"
dtype = torch.float64
from sklearn.datasets import load_svmlight_file
from utils import convert_coo_torch

# path_dataset = "./data/logistic/news20.binary.bz2"
# X,y = load_svmlight_file(path_dataset)
# X = X.tocoo()
# X = convert_coo_torch(X)
# y = torch.from_numpy(y)
# X = X.to(device).to(dtype)
# y = y.to(device).to(dtype)
# y = (y+1)/2
# data_num,dim = X.shape

data_num = 100
dim = 1000000

X = torch.randn(data_num,dim,device=device,dtype=dtype)
w = torch.randn(dim,device=device,dtype=dtype)
y = X@w
y+= torch.randn(data_num)/1000
y[y>=0]= 1
y[y<0] = 0

torch.save(X,"X.pth")
torch.save(y,"y.pth")
# delta = 1
# inner_iteration = 100000
# subproblem_eps = 1e-7

# reduced_dim = 100

# f = robust_logistic(params=[X,y],
#                     delta=delta,
#                     inner_iteration=inner_iteration,
#                     subproblem_eps=subproblem_eps)



iteration = 1

# print("run projection")
# f.projection = True
# u = torch.randn(reduced_dim,device=device,dtype=dtype)
# torch.cuda.synchronize()
# start = time.time()
# for i in range(iteration):
#     x = torch.randn(dim,device = device,dtype=dtype)
#     print(f(x,u))
# torch.cuda.synchronize()
# print(time.time() - start)

# print("run func")
# f.projection = False
# torch.cuda.synchronize()
# start = time.time()
# for i in range(iteration):
#     x = torch.randn(dim,device = device,dtype=dtype)
#     print(f(x))
# torch.cuda.synchronize()
# print(time.time() - start)

# u = torch.randn(reduced_dim,device=device,dtype=dtype)
# for i in range(iteration):
#     x = torch.randn(dim,device = device,dtype=dtype)
#     f.projection = False
#     x_original = f.get_subproblem_solution(x)
#     print(f(x))
#     print("original",torch.linalg.norm(x_original))
    
#     f.projection = True
#     x_projection = f.get_subproblem_solution(x,u)
#     print(f(x,u))
#     print("projection",torch.linalg.norm(x_projection))
#     P = torch.randn(reduced_dim,dim,device = device,dtype=dtype)/(dim**0.5)
#     P = torch.concat((x.unsqueeze(0),P),dim=0)
#     x_projection_dim = torch.linalg.lstsq(P,x_projection).solution
#     print("projection_dim",torch.linalg.norm(x_projection_dim))
    
