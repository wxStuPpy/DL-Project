import torch
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split

import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入自定义组件
from project.ITH.common import utils
from denoising_config import *
from denoising_data import ImageDataSet
from denoising_model import *


def test(denoiser, test_loader, device):
    denoiser.eval()
    # 从DataLoader中取出一个批次
    data_iter = iter(test_loader)
    noise_imgs, original_imgs = next(data_iter)
    print(f'Noise Image shape: {noise_imgs.shape}')

    denoiser = denoiser.to(device)
    noise_imgs = noise_imgs.to(device)
    outputs = denoiser(noise_imgs)
    print(f'output shape: {outputs.shape}')

    # 变化数据 准备绘图
    # 1.噪声图像
    noise_imgs = noise_imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
    print('noise_imgs.shape', noise_imgs.shape)
    # 2.去噪图像
    outputs = outputs.permute(0, 2, 3, 1).detach().cpu().numpy()
    print('outputs.shape', outputs.shape)
    # 3.原始图像
    original_imgs = original_imgs.permute(0, 2, 3, 1).cpu().numpy()
    print('original_imgs.shape', original_imgs.shape)
    fig, axes = plt.subplots(3, 10, figsize=(25, 4))
    for imgs, row in zip([noise_imgs, outputs, original_imgs], axes):
        for img, ax in zip(imgs, row):
            ax.imshow(img)
            ax.axis('off')
    plt.show()

if __name__ == '__main__':
    #环境配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        print(f"当前设备 ID: {torch.cuda.current_device()}")
    else:
        print("⚠️ 当前为 CPU 运行！")

    # 设置随机种子
    utils.seed_everything(SEED)
    #数据加载
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])
    dataset = ImageDataSet(IMG_PATH, transform=transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_SIZE, TEST_SIZE])

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
    )

    #模型
    loaded_denoiser=ConvDenoiser()
    model_state_dict=torch.load(DENOISER_MODEL_NAME,map_location=device)
    loaded_denoiser.load_state_dict(model_state_dict)
    loaded_denoiser.to(device)

    #测试
    test(loaded_denoiser, test_loader, device)