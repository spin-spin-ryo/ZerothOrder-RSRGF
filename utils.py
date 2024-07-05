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
      column_index = torch.arange(n,device = DEVICE).repeat(s).reshape(1,-1)
   if prob_vector is None:
      prob_vector = 0.5*torch.ones(s*n,device = DEVICE,dtype = DTYPE)
   row_index = torch.randint(0,d,(1,n*s),device = DEVICE)
   index = torch.cat((column_index,row_index),dim=0)
   values = (2*torch.bernoulli(prob_vector)-1)/s**0.5
   return torch.sparse_coo_tensor(indices = index, values = values,device = DEVICE)

def generate_sub_orthogonal(reduced_dim,dim):
   v = torch.randn(1,dim,device = DEVICE)
   v/= torch.linalg.norm(v)
   a = v.transpose(0,1)[:reduced_dim]
   G = torch.eye(reduced_dim,dim,device = DEVICE)-2*a@v
   return G


def modifying_parameters(solver_name,reduced_dims,heuristic_intervals,sparsity,projection):
   if solver_name == "RGF":
      reduced_dims = [None]
      heuristic_intervals = [None]
      sparsity = None
      projection = None
   elif solver_name == "proposed":
      if projection is None:
         projection = False
   
   return reduced_dims,heuristic_intervals,sparsity,projection
            
def generate_sparse_random_matrix(data_num, feature_num, sparsity=0.1):
    """
    Generates a sparse random matrix of size (data_num, feature_num).

    Parameters:
    - data_num (int): Number of rows of the matrix.
    - feature_num (int): Number of columns of the matrix.
    - sparsity (float): Proportion of elements that are non-zero (default is 0.1).

    Returns:
    - sparse_matrix (torch.sparse.FloatTensor): A sparse random matrix.
    """
    # Calculate the number of non-zero elements
    num_nonzeros = int(data_num * feature_num * sparsity)
    
    # Generate random indices for non-zero elements
    indices = torch.randint(0, data_num, (2, num_nonzeros))
    indices[1] = torch.randint(0, feature_num, (num_nonzeros,))
    
    # Generate random values for the non-zero elements
    values = torch.randn(num_nonzeros)
    
    # Create the sparse matrix
    sparse_matrix = torch.sparse.FloatTensor(indices, values, torch.Size([data_num, feature_num]))
    
    return sparse_matrix