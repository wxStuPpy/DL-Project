__all__=['ConvEncoder','ConvDecoder']

import torch
import torch.nn as nn

#定义编码器类
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        #五层卷积池化
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        #五层转置卷积
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2,stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2,stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2,stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2,stride=2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2,stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    x=torch.randn(1,3,64,64)
    y=encoder(x)
    print(y.shape)
    z=decoder(y)
    print(z.shape)
