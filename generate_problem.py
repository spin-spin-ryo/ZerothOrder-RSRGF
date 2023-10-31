import torch
import os
from environments import DATAPATH
from utils import *
from function import *
import pickle


def generate(mode,properties):
    if mode == "test":
        f ,x0 = generate_test(properties)
    elif mode == "Quadratic":
        f,x0 = generate_quadratic(properties)
    elif mode == "max-linear":
        f,x0 = generate_max_linear(properties)
    elif mode == "piecewise-linear":
        dim = int(properties["dim"])
        f = piecewise_linear()
        x0 = torch.zeros(dim)
    elif mode == "subspace-norm":
        f,x0 = generate_subspace(properties)
    elif mode == "subspace-norm local":
        f,x0 = generate_subspace(properties,local = True)
    elif mode == "LinearRegression":
        f,x0 = generate_LinearRegression(properties)
    elif mode == "NMF":
        f,x0 = generate_nmf(properties)
    elif mode == "softmax":
        f,x0 = generate_softmax(properties)
    elif mode == "regularized softmax":
        f,x0 = generate_softmax(properties)
        f = generate_regularized(f,properties)
    elif mode == "regularized LinearRegression":
        f,x0 = generate_LinearRegression(properties)
        f = generate_regularized(f,properties)
    elif mode == "regularized NMF":
        f,x0 = generate_nmf(properties)
        f = generate_regularized(f,properties)
    elif mode == "regularized test":
        f ,x0 = generate_test(properties)
        f = generate_regularized(f,properties)
    elif mode == "regularized test local":
        f ,x0 = generate_test(properties,local = True)
        f = generate_regularized(f,properties)
    elif mode == "regularized Quadratic":
        f,x0 = generate_quadratic(properties)
        f = generate_regularized(f,properties)
    elif mode == "regularized Logistic":
        f,x0 = generate_logistic(properties)
        f = generate_regularized(f,properties)
    elif mode == "regularized LinearRegression local":
        f,x0 = generate_LinearRegression(properties,local = True)
        f = generate_regularized(f,properties)
    elif mode == "regularized NMF local":
        f,x0 = generate_nmf(properties,local = True)
        f = generate_regularized(f,properties)
    elif mode == "adverserial attack":
        f,x0 = generate_adverserial(properties)
    else:
        raise ValueError("No functions.")
    return f,x0

def generate_max_linear(properties):
    dim = int(properties["dim"])
    number = int(properties["number"])
    savepath = os.path.join(DATAPATH,"max-linear")
    filename_A = f"A_{dim}_{number}.pth"
    filename_b = f"b_{dim}_{number}.pth"
    filename_x0 = f"x0_{dim}_{number}.pth"
    if os.path.exists(os.path.join(savepath,filename_A)):
        A = torch.load(os.path.join(savepath,filename_A))
        b = torch.load(os.path.join(savepath,filename_b))
        x0 = torch.load(os.path.join(savepath,filename_x0))
    else:
        os.makedirs(savepath,exist_ok=True)
        A = torch.randn(number,dim)
        b = torch.ones(number)
        x0 = torch.randn(dim)*10
        torch.save(A,os.path.join(savepath,filename_A))
        torch.save(b,os.path.join(savepath,filename_b))
        torch.save(x0,os.path.join(savepath,filename_x0))
    params = [A,b]
    f = max_linear(params)
    return f,x0


    

def generate_test(properties,local = False):
    dim = int(properties["dim"])
    if not local:
        x0 = torch.ones(dim)
    else:
        x0 = torch.ones(dim) / dim
    f = test_function()
    return f,x0

def generate_subspace(properties,local = False):
    dim = int(properties["dim"])
    subspace_dim = int(properties["subspace"])
    ord = int(properties["ord"])
    f = subspace_norm([torch.tensor(subspace_dim),torch.tensor(ord)])
    if not local:
        x0 = torch.ones(dim)
    else:
        x0 = torch.ones(dim)/dim
    return f,x0

def generate_quadratic(properties):
    property = properties["property"] 
    dim = int(properties["dim"])
    rank = int(properties["rank"])
    savepath = os.path.join(DATAPATH,"quadratic",property)
    filename_Q = f"Q_{dim}_{rank}.pth"
    filename_b = f"b_{dim}_{rank}.pth"
    filename_x0 = f"x0_{dim}_{rank}.pth"
    if os.path.exists(os.path.join(savepath,filename_Q)):
        Q = torch.load(os.path.join(savepath,filename_Q))
        b = torch.load(os.path.join(savepath,filename_b))
        x0 = torch.load(os.path.join(savepath,filename_x0))
        
    else:
        os.makedirs(savepath,exist_ok=True)
        if property == "convex":
            Q = generate_semidefinite(dim,rank)
            b = torch.randn(dim)
            x0 = 10*torch.randn(dim)
        elif property == "sconvex":
            Q = generate_definite(dim)
            b = torch.randn(dim)
            x0 = 10*torch.randn(dim)
        elif property == "nonconvex":
            Q = generate_symmetric(dim)
            b = torch.randn(dim)  
            x0 = 10*torch.randn(dim)  
        else:
            raise ValueError("There is no property.")
        torch.save(Q,os.path.join(savepath,filename_Q))
        torch.save(b,os.path.join(savepath,filename_b))
        torch.save(x0,os.path.join(savepath,filename_x0))
    params = [Q,b]
    f = QuadraticFunction(params=params)
    return f,x0

def generate_logistic(properties):
    dim = properties["dim"]
    data_num = properties["data-num"]
    savepath = os.path.join(DATAPATH,"logistic")
    filename_A = f"A_{dim}_{data_num}.pth"
    filename_b = f"b_{dim}_{data_num}.pth"
    filename_x0 = f"x0_{dim}_{data_num}.pth"
    if os.path.exists(os.path.join(savepath,filename_A)):
        A = torch.load(os.path.join(savepath,filename_A))
        b = torch.load(os.path.join(savepath,filename_b))
        x0 = torch.load(os.path.join(savepath,filename_x0))
    else:
        os.makedirs(savepath,exist_ok=True)
        A = torch.randn(data_num,dim)
        b = generate_zeroone(data_num)
        x0 = torch.randn(dim)*10
        torch.save(A,os.path.join(savepath,filename_A))
        torch.save(b,os.path.join(savepath,filename_b))
        torch.save(x0,os.path.join(savepath,filename_x0))
    params = [A,b]
    f = logistic(params)
    return f,x0

def generate_regularized(f,properties):
    p = torch.tensor(properties["ord"]).to(torch.int32)
    coef = torch.tensor(properties["coef"])
    fused_flag = properties["fused"]
    
    if fused_flag:
        dim = properties["dim"]
        A = generate_fusedmatrix(dim)
    else:
        A = None
    params = [p,coef,A]
    f_r = regularizedfunction(f,params)
    return f_r
    
def generate_LinearRegression(properties,local = False):
    data_name = properties["data-name"]
    bias = bool(properties["bias"])
    if data_name == "E2006":
        from sklearn.datasets import load_svmlight_file
        from utils import convert_coo_torch
        path_dataset = "./data/LinearRegression/E2006.train.bz2"
        X,y = load_svmlight_file(path_dataset)
        X = X.tocoo()
        X = convert_coo_torch(X)
        y = torch.from_numpy(y)
        dim = X.shape[1]
        if bias:
            dim += 1
        data_num = X.shape[0]
        if not local:
            x0 = torch.zeros(dim)
        else:
            path_local_init = "./data/LinearRegression/E2006_init.pth"
            x0 = torch.load(path_local_init)
    elif data_name == "random-100-10000":
        dim = 10000
        if bias:
            dim += 1
        X = torch.load("./data/LinearRegression/random-100-10000-X.pth")
        y = torch.load("./data/LinearRegression/random-100-10000-y.pth")
        x0 = torch.zeros(dim)
    
    elif data_name == "random-100-100000":
        dim = 100000
        if bias:
            dim += 1
        X = torch.load("./data/LinearRegression/random-100-100000-X.pth")
        y = torch.load("./data/LinearRegression/random-100-100000-y.pth")
        x0 = torch.zeros(dim)
        
    else:
        raise ValueError("No matching data name.")

    params = [X,y]
    f = LinearRegression(params=params,bias=bias)
    return f,x0

def generate_nmf(properties,local = False):
    data_name = properties["data-name"]
    rank = properties["rank"]
    if data_name == "movie":
        path_dataset = "./data/NMF/movie_100k.pth"
        with open(path_dataset,"rb") as data:
            W = pickle.load(data)
        data_num,feature_num = W.shape
    elif data_name == "random":
        data_num = properties["data-num"]
        feature_num = properties["feature-num"]
        W = torch.randn(data,feature_num)
    
    dim = data_num*rank + feature_num*rank
    if not local:
        x0 = torch.ones(dim)
    else:
        nmf_init_path = "./data/NMF/solution_disturb.pth"
        x0 = torch.load(nmf_init_path)
    
    params = [W,torch.tensor(rank)]
    f = NMF(params=params)
    return f,x0

def generate_softmax(properties):
    data_name = properties["data-name"]
    if data_name == "Scotus":
        from sklearn.datasets import load_svmlight_file
        from utils import convert_coo_torch
        path_dataset = "./data/logistic/scotus_lexglue_tfidf_train.svm.bz2"
        X,y = load_svmlight_file(path_dataset)
        X = X.tocoo()
        X = convert_coo_torch(X)
        y = torch.from_numpy(y)
        y = y.to(torch.int64)
        y=F.one_hot(y)
        data_num,feature_num = X.shape
        _,class_num = y.shape
        dim = feature_num*class_num + class_num
        x0 = torch.zeros(dim)
    elif data_name == "news20":
        from sklearn.datasets import load_svmlight_file
        from utils import convert_coo_torch
        path_dataset = "./data/logistic/news20.bz2"
        X,y = load_svmlight_file(path_dataset)
        X = X.tocoo()
        X = convert_coo_torch(X)
        y = torch.from_numpy(y)
        y = y.to(torch.int64)
        y=F.one_hot(y)
        data_num,feature_num = X.shape
        _,class_num = y.shape
        dim = feature_num*class_num + class_num
        x0 = torch.zeros(dim)
    elif data_name == "random":
        return
    params = [X,y]
    f = softmax(params)
    return f,x0

def generate_adverserial(properties):
    data_name = properties["data-name"]
    epoch_num = int(properties["epoch-num"])
    coef = torch.tensor(properties["coef"])
    if data_name == "Scotus":
        from sklearn.datasets import load_svmlight_file
        from utils import convert_coo_torch
        path_dataset = "./data/logistic/scotus_lexglue_tfidf_train.svm.bz2"
        X,y = load_svmlight_file(path_dataset)
        X = X.tocoo()
        X = convert_coo_torch(X)
        X = X.to_dense()
        y = torch.from_numpy(y)
        y = y.to(torch.int64)

    elif data_name == "news20":
        from sklearn.datasets import load_svmlight_file
        from utils import convert_coo_torch
        path_dataset = "./data/logistic/news20.bz2"
        X,y = load_svmlight_file(path_dataset)
        X = X.tocoo()
        X = convert_coo_torch(X)
        X = X.to_dense()
        y = torch.from_numpy(y)
        y = y.to(torch.int64)
    
    params = [X,y,epoch_num,coef]
    features_num = X.shape[1]
    f = adversarial(params=params)
    x0 = torch.ones(features_num)
    return f,x0

    
