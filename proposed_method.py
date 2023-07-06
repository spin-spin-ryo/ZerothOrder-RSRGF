from optim_method import __optim__
import torch

class proposed(__optim__):
    def __init__(self,determine_stepsize):
        #params = [reduced_dim,sample_size,mu,lr]
        self.determine_stepsize  = determine_stepsize
        super().__init__()
    
    def __direction__(self):
        with torch.no_grad():
            reduced_dim = self.params[0]
            sample_size = self.params[1]
            mu = self.params[2]
            dim = self.xk.shape[0]
            P = torch.randn(dim,reduced_dim,device = self.device,dtype = self.dtype)/torch.sqrt(torch.tensor(reduced_dim,device = self.device,dtype = self.dtype))

            subspace_func = lambda d: self.func(self.xk + P@d)

            subspace_dir = None
            for i in range(sample_size):
                d = torch.randn(reduced_dim,device= self.device,dtype = self.dtype)/torch.sqrt(torch.tensor(sample_size,device = self.device,dtype = self.dtype))
                g1 = subspace_func(mu*d)
                if subspace_dir is None:
                    subspace_dir = (g1 - self.loss.item())/mu * d
                else:
                    subspace_dir += (g1 - self.loss.item())/mu * d
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

    