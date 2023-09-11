import torch

X = torch.randn(100,100000)
y = torch.randn(100)

torch.save(X,"./data/LinearRegression/random-100-100000-X.pth")
torch.save(y,"./data/LinearRegression/random-100-100000-y.pth")