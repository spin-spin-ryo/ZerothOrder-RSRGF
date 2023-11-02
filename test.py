from solver import BackTrackingPGD,projection_ball2,BackTrackingAccerelatedPGD
import torch

dim = 10000
A = torch.randn(1000,dim)
b = torch.randn(1000)

r = 0.1

def func(x):
    return torch.logsumexp(A@x+b,dim = 0)

def prox(x,t):
    return projection_ball2(x=x,t=t,r=r)

solver1 = BackTrackingPGD(func=func,prox=prox)
solver2 = BackTrackingAccerelatedPGD(func=func,prox=prox)

x = torch.randn(dim)
iteration = 100
solver1.__iter__(x0=x,
                 iteration=iteration)

print(solver1.get_function_value())
print(solver1.get_iteration())
print(solver1.get_grad_norm())

# solver2.__iter__(x0=x,
#                  iteration=iteration)

# print(solver2.get_function_value())
# print(solver2.get_iteration())
# print(solver2.get_grad_norm())

