__all__ = ['ImageLabelDataSet']

import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import re
import torchvision.transforms as T
from classification_config import *


# 按字母数字混合排序
def sorted_alphanum(img_names):
    # 转化函数
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda image_name: [convert(x) for x in re.split('([0-9]+)', image_name)]
    return sorted(img_names, key=alphanum_key)


# 自定义数据集类型
class ImageLabelDataSet(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(self.main_dir))
        self.labels = pd.read_csv(label_path)  # 读取分类标签文件 得到DataFrame
        self.label_dict = dict(zip(self.labels['id'], self.labels['target']))  # 分离id和标签 得到字典

    def __len__(self):
        return len(self.image_names)

    # 传入图片ID 获取数据集元素(X,y)  不是一次性加载 选择性加载 防止爆内存
    def __getitem__(self, idx):
        # 1.根据索引号 获得图片完整路径
        image_loc = os.path.join(self.main_dir, self.image_names[idx])
        # 2.使用PIL打开图片
        image = Image.open(image_loc).convert('RGB')
        # 3.利用transform转化
        if self.transform:
            tensor_image = self.transform(image)
        else:
            raise ValueError('transform is not defined')
        # 在字典中找到标签
        label = self.label_dict[idx]
        return tensor_image, label

if __name__ == '__main__':
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
    ])
    dataset=ImageLabelDataSet(image_dir=IMG_PATH, label_path=LABELS_PATH,transform=transform)
    print(len(dataset))