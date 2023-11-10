import torch
from optim_method import logger

def projection_ball2(x,t,r = 1):
    # \|x\|_2 \le rへの射影
    if x@x <= r**2:
        return x
    else:
        return r*x/torch.linalg.norm(x)
    

class BackTrackingPGD:
    def __init__(self,func,prox):
        self.func = func
        self.xk = None
        self.prox = prox
        self.iter = 0
        # prox(x,t)
    
    def backtracking(self,g,beta,eps = 0):
        max_iter = 10000
        t = 1
        x_ = self.prox(self.xk - t*g,t)
        while t*self.func(x_) > t*self.func(self.xk) - t*g@(self.xk - x_) + 1/2*((self.xk - x_)@(self.xk - x_)) + eps:
            t *= beta
            max_iter -= 1
            x_ = self.prox(self.xk - t*g,t)
            if max_iter < 0:
                logger.info("error")
                logger.info(self.func(x_))
                break
        return t

    def __one_iter__(self,eps,beta):
        self.iter +=1
        self.xk.grad = None
        f = self.func(self.xk)
        f.backward()
        g = self.xk.grad
        with torch.no_grad():
            t = self.backtracking(g,beta=beta)
            x_ = self.prox(self.xk - t*g,t)
            if self.check_stop(x_ - self.xk,t*eps):
                return True
            self.xk = x_.detach().clone()
            self.xk.requires_grad = True        
        return False

    
    def check_stop(self,G_t,eps):
        return torch.linalg.norm(G_t) < eps

    def __iter__(self,x0,iteration,eps=1e-6,beta = 0.8):
        self.iter = 0
        self.xk = x0.clone().detach()
        self.xk.requires_grad = True
        for idx in range(iteration):
            stop_flag = self.__one_iter__(eps=eps,beta=beta)
            if stop_flag:
                return    
        print("max iteration")
    
    def get_function_value(self):
        with torch.no_grad():
            return self.func(self.xk)
    
    def get_solution(self):
        return self.xk.clone().detach()

    def get_grad_norm(self,beta = 0.8):
        self.xk.grad = None
        f = self.func(self.xk)
        f.backward()
        g = self.xk.grad
        with torch.no_grad():
            t = self.backtracking(g,beta=beta)
            G_t = (self.xk - self.prox(self.xk - t*g,t))/t
            return torch.linalg.norm(G_t)
        
        
    def get_iteration(self):
        return self.iter

class BackTrackingAccerelatedPGD(BackTrackingPGD):
    def __init__(self, func, prox):
        super().__init__(func, prox)
        self.t = 1
        self.v = None
        self.xk1 = None
        self.k = 0

    def __iter__(self,x0,iteration,eps=1e-6,beta = 0.8,restart = False):
        self.iter = 0
        self.k = 0
        self.xk = x0.clone().detach()
        self.xk1 = x0.clone().detach()
        for idx in range(iteration):
            stop_flag = self.__one_iter__(eps=eps,beta=beta,restart=restart)
            if stop_flag:
                return    
        print("max iteration")
    
    
    def backtracking(self, g, beta):
        _x = self.prox(self.v-self.t*g,self.t)

        while self.t*self.func(_x) > self.t*self.func(self.v) + self.t*g@(_x - self.v) + 1/2*((_x - self.v)@(_x - self.v)):
            self.t *= beta
            _x = self.prox(self.v-self.t*g,self.t)    
        return self.t

    def __one_iter__(self,eps,beta,restart):
        self.iter +=1
        self.k += 1
        k = self.k
        self.v = self.xk + (k-2)/(k+1)*(self.xk - self.xk1)
        self.v.requires_grad = True
        f = self.func(self.v)
        f.backward()
        g = self.v.grad
        with torch.no_grad():
            t = self.backtracking(g,beta=beta)
            x_ = self.prox(self.v - t*g,t)
            if self.check_stop(x_ - self.v,t*eps):
                return True
            self.xk1 = self.xk
            self.xk = x_.detach().clone()
            self.v = None
            if restart:
                if self.func(self.xk) > self.func(self.xk1):
                    self.k = 0      
        return False

    def get_grad_norm(self,beta = 0.8):
        self.xk.requires_grad = True
        self.xk.grad = None
        f = self.func(self.xk)
        f.backward()
        g = self.xk.grad
        with torch.no_grad():
            self.v = self.xk
            t = self.backtracking(g,beta=beta)
            G_t = (self.xk - self.prox(self.xk - t*g,t))/t
            self.v = None
        self.xk.requires_grad = False
        return torch.linalg.norm(G_t)
        
    