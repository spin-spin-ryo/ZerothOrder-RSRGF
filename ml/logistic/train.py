import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,Dataset
import numpy as np

def convert_coo_torch(X):
    values = X.data
    indices = np.vstack((X.row,X.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = X.shape
    return torch.sparse.FloatTensor(i,v,torch.Size(shape))


def get_dataset(path_dataset = "./data/logistic/scotus_lexglue_tfidf_train.svm.bz2"):
    from sklearn.datasets import load_svmlight_file
    X,y = load_svmlight_file(path_dataset)
    X = X.tocoo()
    X = convert_coo_torch(X)
    y = torch.from_numpy(y)
    y = y.to(torch.int64)
    class_num = torch.max(y)+1
    return TensorDataset(X.to(device),y.to(device)),class_num

def get_dataloader(dataset,batch_size):
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

def get_model(feartures_num,class_num):
    from models import Logistic
    model = Logistic(features_num=feartures_num,class_num=class_num)
    return model

def get_loss():
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn

def get_optimizer(model,lr,**kwargs):
    optimizer = optim.Adam(model.parameters(),lr = lr,**kwargs)
    return optimizer

def train_model_one_epoch(model,Loss,optimizer,data_loader,validation_iter = 1):
    sum_loss = 0
    total_iter = len(data_loader)
    l = 1e-6
    

    for idx, bacth in enumerate(data_loader):
        X,y = bacth
        optimizer.zero_grad()
        pred = model(X)
        loss = Loss(pred,y)
        l1 = torch.tensor(0., requires_grad=True,device=device)
        for w in model.parameters():
            l1 = l1 + torch.linalg.norm(w, 1)
        loss = loss + l*l1
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        if (idx + 1)%validation_iter == 0:
            print(f"Iter {idx+1}/{total_iter} Loss: {sum_loss/(idx+1)}")
        

def save_model(model,epoch_num,save_path = ".",optimizer = None):
    model_file_name = f"model_epoch{epoch_num}.pth"

    torch.save(model.state_dict(),os.path.join(save_path,model_file_name))
    if optimizer is not None:
        optimizer_file_name = f"optimizer_epoch{epoch_num}.pth"
        torch.save(optimizer.state_dict(),os.path.join(save_path,optimizer_file_name))
    
    return


def train(epoch_num,batch_size,lr,data_path,save_dir,checkpoints = 1):
    dataset,class_num = get_dataset(path_dataset=data_path)
    X,y = dataset[0]
    features_num = X.shape[0]
    data_loader = get_dataloader(dataset=dataset,batch_size=batch_size)
    model = get_model(feartures_num=features_num,class_num=class_num)
    model = model.to(device)
    Loss = get_loss()
    optimizer = get_optimizer(model=model,lr=lr)


    for idx in range(epoch_num):
        print(f"epoch {idx+1}")
        train_model_one_epoch(model=model,
                              Loss=Loss,
                              optimizer=optimizer,
                              data_loader=data_loader)
        if idx%checkpoints == 0:
            save_model(model=model,
                    epoch_num=idx,
                    save_path=save_dir,
                    optimizer=optimizer)

if __name__ == "__main__":
    dataset_path = "./dataset/scotus_lexglue_tfidf_train.svm.bz2"
    save_path = "./checkpoints"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # device = "cpu"
    epoch_num = 100000
    batch_size = 6400
    lr = 0.1
    print("start train")
    train(epoch_num,batch_size,lr,data_path=dataset_path,save_dir=save_path,checkpoints=100)
    

        



