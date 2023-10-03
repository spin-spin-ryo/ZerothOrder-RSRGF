import torch
from torch.autograd.functional import hvp
from environments import DEVICE,DTYPE
import numpy as np

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

def compute_hvp(func,x,v):
   return hvp(func,(x,),(v,))[1][0]


def generate_semidefinite(dim,rank):
   P = torch.randn(dim,rank,device = DEVICE)
   return P@P.transpose(0,1)

def generate_symmetric(dim):
   P = torch.randn(dim,dim,device = DEVICE)
   return (P + P.transpose(0,1))/2

def generate_definite(dim):
   P = generate_symmetric(dim)
   min_eig = GetMinimumEig(P)
   if min_eig < 0:
      return P - 10*min_eig*torch.eye(dim,device= DEVICE)
   else:
      return P
 
def generate_zeroone(dim):
   a = torch.randn(dim)
   a[a>=0] = 1
   a[a<0] = -1
   a = a.to(torch.int64)
   return a

def generate_fusedmatrix(dim):
   def fused_func(x):
      return x[:-1] - x[1:]
   return fused_func

def convert_coo_torch(X):
    values = X.data
    indices = np.vstack((X.row,X.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = X.shape
    return torch.sparse.FloatTensor(i,v,torch.Size(shape))

def generate_sparse_random(d,n,s,column_index = None,prob_vector = None):
   if isinstance(s,int):
      pass
   elif isinstance(s,float):
      s = int(d*s)
   if column_index is None:
      column_index = torch.arange(n+1,device = DEVICE)*d
   if prob_vector is None:
      prob_vector = 0.5*torch.ones(s*n,device = DEVICE,dtype = DTYPE)
  
   row_index = torch.randint(0, d, (n * s,),device = DEVICE)
   values = (2*torch.bernoulli(prob_vector)-1)/s**0.5
   return torch.sparse_csr_tensor(crow_indices=column_index,col_indices=row_index, values = values)