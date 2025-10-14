import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入自定义组件
from project.ITH.common import utils
from denoising_config import *
from denoising_data import ImageDataSet
from denoising_model import *
from denoising_engine import *
#from denoising_test import test

def print_gpu_usage():
    """打印当前 GPU 的显存占用情况"""
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2
        used_mem = torch.cuda.memory_allocated(gpu_id) / 1024**2
        cached_mem = torch.cuda.memory_reserved(gpu_id) / 1024**2
        print(f"💻 GPU 显存使用情况: 已分配 {used_mem:.1f}MB / 已缓存 {cached_mem:.1f}MB / 总显存 {total_mem:.1f}MB")
    else:
        print("⚠️ 当前设备为 CPU，无 GPU 信息。")

if __name__ == '__main__':
    # 1️⃣环境配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 当前使用设备: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA 是否可用: {torch.cuda.is_available()}")
        print(f"✅ 当前设备 ID: {torch.cuda.current_device()}")
    else:
        print("⚠️ 当前为 CPU 运行！")

    # 设置随机种子
    utils.seed_everything(SEED)
    # 2️⃣ 数据加载
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])
    dataset = ImageDataSet(IMG_PATH, transform=transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_SIZE, TEST_SIZE])
    print(f"✅ 数据集加载完成：训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张")

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
    )

    # 3️⃣ 模型与优化器
    denoiser = ConvDenoiser()
    denoiser.to(device)
    loss = nn.MSELoss()
    optimizer = optim.Adam(denoiser.parameters(), lr=LEARNING_RATE)
    print(f"✅ 模型加载完成，学习率: {LEARNING_RATE}")

    # 初始化最小测试误差 以找到最优模型
    min_test_loss = np.inf

    # 4️⃣ 训练与测试循环
    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        tqdm.write("\n" + "=" * 50)
        tqdm.write(f"🧩 Epoch [{epoch + 1}/{EPOCHS}] begin...")

        # 训练一轮
        train_loss = train_step(denoiser, train_loader, loss, optimizer, device)
        tqdm.write(f"📈 Epoch [{epoch + 1}/{EPOCHS}]  - Train Loss: {train_loss:.6f}")

        # 测试一轮
        test_loss = test_step(denoiser, test_loader, loss, device)
        tqdm.write(f"🧪 Epoch [{epoch + 1}/{EPOCHS}]  - Test Loss: {test_loss:.6f}")

        # 判断并保存模型
        if test_loss < min_test_loss:
            tqdm.write(f" descend: {min_test_loss:.6f} → {test_loss:.6f}")
            tqdm.write(" saving model...")
            min_test_loss = test_loss
            torch.save(denoiser.state_dict(), DENOISER_MODEL_NAME)
        else:
           tqdm.write('not saving model')
        print_gpu_usage()

    # 6️⃣ 训练结束
    print(f"best loss: {min_test_loss:.6f}")

    #测试
    # loaded_denoiser = ConvDenoiser()
    # model_state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    # loaded_denoiser.load_state_dict(model_state_dict)
    # loaded_denoiser.to(device)
    #
    # # 测试
    # test(loaded_denoiser, test_loader, device)

