from optim_method import __optim__
from utils import generate_sparse_random
from environments import DEVICE,DTYPE
import torch
import time



class proposed(__optim__):
    def __init__(self,determine_stepsize,central = False,projection = False):
        #params = [reduced_dim,sample_size,mu,lr]
        self.determine_stepsize  = determine_stepsize
        self.central = central
        super().__init__()
        self.projection = projection
        print("central",self.central)
    
    def __direction__(self,loss):
        reduced_dim = self.params[0]
        sample_size = self.params[1]
        mu = self.params[2]
        dim = self.xk.shape[0]
        P = torch.randn(dim,reduced_dim,device = self.device,dtype = self.dtype)/torch.sqrt(torch.tensor(dim,device = self.device,dtype = self.dtype))
        subspace_dir = None
        if self.projection:
            U = torch.randn(sample_size,reduced_dim+1,device = self.device,dtype = self.dtype)/torch.sqrt(torch.tensor(sample_size,device = self.device,dtype = self.dtype))
            U[:,0]=1
        else:
            U = torch.randn(sample_size,reduced_dim,device = self.device,dtype = self.dtype)/torch.sqrt(torch.tensor(sample_size,device = self.device,dtype = self.dtype))
        if self.central:
            if self.projection:
                for i in range(sample_size):
                    m = mu*P@U[i,1:]
                    g1 = self.func(self.xk + m,u= mu*U[i])
                    g2 = self.func(self.xk - m,u= mu*U[i])
                    if subspace_dir is None:
                        subspace_dir = (g1 - g2)/(2*mu) * U[i,1:]
                    else:
                        subspace_dir += (g1 - g2)/(2*mu) * U[i,1:]
            else:
                for i in range(sample_size):
                    m = mu*P@U[i]
                    g1 = self.func(self.xk + m)
                    g2 = self.func(self.xk - m)
                    if subspace_dir is None:
                        subspace_dir = (g1 - g2)/(2*mu) * U[i]
                    else:
                        subspace_dir += (g1 - g2)/(2*mu) * U[i]
        else:
            for i in range(sample_size):
                g1 = self.func(self.xk + m)
                if subspace_dir is None:
                    subspace_dir = (g1 - loss.item())/mu * U[i]
                else:
                    subspace_dir += (g1 - loss.item())/mu * U[i]
        return - P@subspace_dir

    def __step__(self,i):
        if type(self.determine_stepsize) is str:
            lr = self.params[3]
            dim = self.xk.shape[0]
            reduced_dim = self.params[0]
            return lr*((dim)**2)/(reduced_dim**2)
        elif self.determine_stepsize is not None:
            return self.determine_stepsize(i);
        else:
            lr = self.params[3]
            return lr
    
    def __iter_per__(self, i):
        return super().__iter_per__(i)
    
    def __iter__(self,func,x0,params,iterations,savepath,suffix,interval = None):
        if interval is None:
            interval = iterations
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.params = params
        self.xk = x0
        self.xk.requires_grad_(False)
        self.func = func
        if self.projection:
            self.func.projection = self.projection
        self.__save_init__(iterations,fvalues = "min",time_values = "max",norm_dir = "iter")
        for i in range(iterations):
            self.__iter_per__(i)
            if (i+1)%interval == 0:
                self.__save__(savepath=savepath,suffix=suffix,fvalues = "min",time_values = "max")
                self.__log__(i)
    
    def __iter_per__(self,i):
        self.__clear__()
        torch.cuda.synchronize()
        loss_start_time = time.time()
        loss = self.func(self.xk)
        torch.cuda.synchronize()
        self.loss_time += time.time() - loss_start_time
        dk = self.__direction__(loss)
        lr = self.__step__(i)
        self.__update__(lr*dk)
        torch.cuda.synchronize()
        self.__save_value__(i,fvalues = ("min",loss.item()),
                time_values = ("max",time.time() - self.loss_time - self.start_time),
                norm_dir = ("iter",torch.linalg.norm(dk).item()))
        return
 
class proposed_sparse(proposed):
    def __init__(self, determine_stepsize):
        super().__init__(determine_stepsize)
        self.column_index = None
        self.prob_vector = None

    def __direction__(self,loss):
        with torch.no_grad():
            reduced_dim = self.params[0]
            sample_size = self.params[1]
            mu = self.params[2]
            dim = self.xk.shape[0]
            sparsity = self.params[4]
            if self.column_index is None or self.prob_vector is None:
                s = int(sparsity*reduced_dim)
                self.column_index = torch.arange(dim,device = DEVICE).repeat(s).reshape(1,-1)
                self.prob_vector = 0.5*torch.ones(s*dim,device = DEVICE,dtype = DTYPE)
  

            P = generate_sparse_random(d=reduced_dim,n = dim, s = sparsity,column_index=self.column_index,prob_vector=self.prob_vector)
            subspace_func = lambda d: self.func(self.xk + P@d)
            subspace_dir = None
            U = torch.randn(sample_size,reduced_dim,device = self.device,dtype = self.dtype)/torch.sqrt(torch.tensor(sample_size,device = self.device,dtype = self.dtype))
            for i in range(sample_size):
                g1 = subspace_func(mu*U[i])
                if subspace_dir is None:
                    subspace_dir = (g1 - loss.item())/mu * U[i]
                else:
                    subspace_dir += (g1 - loss.item())/mu * U[i]
            return - P@subspace_dir
    
class proposed_heuristic(proposed):
    def __init__(self, determine_stepsize):
        super().__init__(determine_stepsize)
        self.P = None
    
    def __iter_per__(self,i):
        self.__clear__()
        loss = self.func(self.xk)
        dk = self.__direction__(loss,iteration = i)
        lr = self.__step__(i)
        self.__update__(lr*dk)
        torch.cuda.synchronize()
        self.__save_value__(i,fvalues = ("min",loss.item()),
                time_values = ("max",time.time() - self.start_time),
                norm_dir = ("iter",torch.linalg.norm(dk).item()))
        return
    

    def __direction__(self,loss,**kwargs):
        with torch.no_grad():
            reduced_dim = self.params[0]
            sample_size = self.params[1]
            mu = self.params[2]
            interval = self.params[4]
            dim = self.xk.shape[0]
            if kwargs["iteration"]%interval == 0:
                self.P = torch.randn(dim,reduced_dim,device = self.device,dtype = self.dtype)/torch.sqrt(torch.tensor(dim,device = self.device,dtype = self.dtype))

            subspace_func = lambda d: self.func(self.xk + self.P@d)
            subspace_dir = None
            U = torch.randn(sample_size,reduced_dim,device = self.device,dtype = self.dtype)/torch.sqrt(torch.tensor(sample_size,device = self.device,dtype = self.dtype))
            for i in range(sample_size):
                g1 = subspace_func(mu*U[i])
                if subspace_dir is None:
                    subspace_dir = (g1 - loss.item())/mu * U[i]
                else:
                    subspace_dir += (g1 - loss.item())/mu * U[i]
            return - self.P@subspace_dir
