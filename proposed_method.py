from optim_method import __optim__
import torch
import time

class proposed(__optim__):
    def __init__(self,determine_stepsize):
        #params = [reduced_dim,sample_size,mu,lr]
        self.determine_stepsize  = determine_stepsize
        super().__init__()
    
    def __direction__(self,loss):
        with torch.no_grad():
            reduced_dim = self.params[0]
            sample_size = self.params[1]
            mu = self.params[2]
            dim = self.xk.shape[0]
            P = torch.randn(dim,reduced_dim,device = self.device,dtype = self.dtype)/torch.sqrt(torch.tensor(dim,device = self.device,dtype = self.dtype))

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
    
    def __step__(self,i):
        if self.determine_stepsize is not None:
            return self.determine_stepsize(i);
        else:
            lr = self.params[3]
            return lr
    
    def __iter_per__(self, i):
        with torch.no_grad():
            return super().__iter_per__(i)
    
    def __iter__(self,func,x0,params,iterations,savepath,interval = None):
        if interval is None:
            interval = iterations
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.params = params
        self.xk = x0
        self.func = func
        self.__save_init__(iterations,fvalues = "min",time_values = "max",norm_dir = "iter")
        for i in range(iterations):
            self.__iter_per__(i)
            if (i+1)%interval == 0:
                self.__save__(savepath)
                self.__log__(i)
    
    def __iter_per__(self,i):
        self.__clear__()
        loss = self.func(self.xk)
        dk = self.__direction__(loss)
        lr = self.__step__(i)
        self.__update__(lr*dk)
        torch.cuda.synchronize()
        self.__save_value__(i,fvalues = ("min",loss.item()),
                time_values = ("max",time.time() - self.start_time),
                norm_dir = ("iter",torch.linalg.norm(dk).item()))
        return
 

   
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
