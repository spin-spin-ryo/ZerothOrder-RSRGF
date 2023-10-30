from ml.logistic.models import Logistic
import torch


feature_num = 1000000
class_num = 20

model = Logistic(features_num=feature_num,class_num=class_num)

x = torch.randn(10,feature_num)

pred = model(x)
print(pred.shape)