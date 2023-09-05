from sklearn.datasets import load_svmlight_file
import torch
import pickle
from utils import convert_coo_torch

path_dataset = "./data/NMF/movie_100k.pth"
with open(path_dataset,"rb") as data:
    U = pickle.load(data)

print(U)