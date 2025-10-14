__all__ = ['ImageDataSet']

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import re

from similarity_config import *


#按字母数字混合排序
def sorted_alphanum(img_names):
    #转化函数
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key=lambda image_name:[convert(x) for x in re.split('([0-9]+)', image_name)]
    return sorted(img_names, key=alphanum_key)

#自定义数据集类型
class ImageDataSet(Dataset):
    def __init__(self, image_dir, transform=None):
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(self.main_dir))

    def __len__(self):
        return len(self.image_names)

    #传入图片ID 获取数据集元素(X,y)  不是一次性加载 选择性加载 防止爆内存
    def __getitem__(self, idx):
        #1.根据索引号 获得图片完整路径
        image_loc = os.path.join(self.main_dir, self.image_names[idx])
        #2.使用PIL打开图片
        image = Image.open(image_loc).convert('RGB')
        #3.利用transform转化
        if self.transform:
            tensor_image = self.transform(image)
        else:
            raise ValueError('transform is not defined')
        return tensor_image,tensor_image

if __name__ == '__main__':
    import  torchvision.transforms as T
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
    ])
    dataset=ImageDataSet(IMG_PATH, transform=transform)
    print(len(dataset))


