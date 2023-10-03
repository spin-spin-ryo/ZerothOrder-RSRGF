from utils import generate_sparse_random
import torch
print(torch.__version__)
d = 100
n = 1000000
s = 0.1
generate_sparse_random(d,n,s)