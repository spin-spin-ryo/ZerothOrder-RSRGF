from function import robust_logistic
import torch
import time

device = "cuda"
dtype = torch.float64
from sklearn.datasets import load_svmlight_file
from utils import convert_coo_torch
path_dataset = "./data/logistic/news20.binary.bz2"
X,y = load_svmlight_file(path_dataset)
X = X.tocoo()
X = convert_coo_torch(X)
y = torch.from_numpy(y)

X = X.to(device).to(dtype)
y = y.to(device).to(torch.int64)
data_num,dim = X.shape

delta = 1e-5
inner_iteration = 100000
subproblem_eps = 1e-3

reduced_dim = 100

f = robust_logistic(params=[X,y],
                    delta=delta,
                    inner_iteration=inner_iteration,
                    subproblem_eps=subproblem_eps)



iteration = 100

# print("run projection")
# f.projection = True
# u = torch.randn(reduced_dim,device=device,dtype=dtype)
# torch.cuda.synchronize()
# start = time.time()
# for i in range(iteration):
#     x = torch.randn(dim,device = device,dtype=dtype)
#     f(x,u)
# torch.cuda.synchronize()
# print(time.time() - start)

print("run func")
f.projection = False
torch.cuda.synchronize()
start = time.time()
for i in range(iteration):
    x = torch.randn(dim,device = device,dtype=dtype)
    f(x)
torch.cuda.synchronize()
print(time.time() - start)

# u = torch.randn(reduced_dim,device=device,dtype=dtype)
# for i in range(iteration):
#     x = torch.randn(dim,device = device,dtype=dtype)
    # f.projection = False
    # x_original = f.get_subproblem_solution(x)
    # print(x_original.shape)
    # print("original",torch.linalg.norm(x_original))
    
    # f.projection = True
    # x_projection = f.get_subproblem_solution(x,u)
    # print(x_projection.shape)
    # print("projection",torch.linalg.norm(x_projection))
    # P = torch.randn(reduced_dim,dim,device = device,dtype=dtype)/(dim**0.5)
    # P = torch.concat((x.unsqueeze(0),P),dim=0)
    # x_projection_dim = torch.linalg.lstsq(P,x_projection).solution
    # print("projection_dim",torch.linalg.norm(x_projection_dim))
    
