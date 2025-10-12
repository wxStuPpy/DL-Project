import torch.nn as nn
import torch
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        #编码器
        #三个卷积层
        self.conv1=nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3=nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        #通用池化层 没有参数 不存在反向传播问题
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        #解码器
        #三个转置卷积层
        self.t_conv1=nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=0,output_padding=0)
        self.t_conv2=nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=0,output_padding=0)
        self.t_conv3=nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0,output_padding=0)
        #普通卷积层
        self.conv_out=nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self,x):
        #编码
        x=self.conv1(x)
        x=torch.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=torch.relu(x)
        x=self.pool(x)

        x=self.conv3(x)
        x=torch.relu(x)
        x=self.pool(x)

        #解码
        x=self.t_conv1(x)
        x=torch.relu(x)

        x=self.t_conv2(x)
        x=torch.relu(x)

        x=self.t_conv3(x)
        x=torch.relu(x)

        #最终卷积
        x=self.conv_out(x)
        x=torch.sigmoid(x)
        return x