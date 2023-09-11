import torch

DATAPATH = "./data"
DTYPE = torch.float64

if torch.cuda.is_available():   
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

CONFIGPATH = "./configs"
RESULTPATH = "./results"
KEYPATH = "/Users/dacapo271/.ssh/ist"
