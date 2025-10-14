__all__=['Classifier']

import torch
import torch.nn as nn
#定义模型
class Classifier(nn.Module):
    def __init__(self,n_classes=5):
        super(Classifier, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=16*16*16, out_features=n_classes),
        )

    def forward(self,x):
        x=self.model(x)
        return x

if __name__=='__main__':
    x=torch.randn(1,3,64,64)
    model=Classifier(n_classes=5)
    y=model(x)
    print(y.shape)