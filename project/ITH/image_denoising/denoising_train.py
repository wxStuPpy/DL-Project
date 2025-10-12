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
from denoising_engine import test_step,train_step

if __name__ == '__main__':
    # 1️⃣ 环境配置
    print("🔧 初始化训练环境...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")

    # 设置随机种子
    utils.seed_everything(SEED)
    print(f"✅ 随机种子已设置: {SEED}")

    # 2️⃣ 数据加载
    print("\n📦 正在加载数据集...")
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])
    dataset = ImageDataSet(IMG_PATH, transform=transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_SIZE, TEST_SIZE])
    print(f"✅ 数据集加载完成：训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张")

    print("\n📂 正在创建 DataLoader...")
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
    print(f"✅ DataLoader 创建成功 - "
          f"train_batch_size: {TRAIN_BATCH_SIZE}, test_batch_size: {TEST_BATCH_SIZE}")

    # 3️⃣ 模型与优化器
    print("\n⚙️ 正在加载模型...")
    denoiser = ConvDenoiser().to(device)
    loss = nn.MSELoss()
    optimizer = optim.Adam(denoiser.parameters(), lr=LEARNING_RATE)
    print(f"✅ 模型加载完成，学习率: {LEARNING_RATE}")

    # 初始化最小测试误差 以找到最优模型
    min_test_loss = np.inf

    print("\n🚀 开始训练！")
    # 4️⃣ 训练与测试循环
    for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
        tqdm.write("\n" + "=" * 50)
        tqdm.write(f"🧩 Epoch [{epoch + 1}/{EPOCHS}] 开始...")

        # 训练一轮
        train_loss = train_step(denoiser, train_loader, loss, optimizer, device)
        tqdm.write(f"📈 Epoch [{epoch + 1}/{EPOCHS}] 训练完成 - Train Loss: {train_loss:.6f}")

        # 测试一轮
        test_loss = test_step(denoiser, test_loader, loss, device)
        tqdm.write(f"🧪 Epoch [{epoch + 1}/{EPOCHS}] 测试完成 - Test Loss: {test_loss:.6f}")

        # 判断并保存模型
        if test_loss < min_test_loss:
            tqdm.write(f" 测试误差下降: {min_test_loss:.6f} → {test_loss:.6f}")
            tqdm.write(" 模型性能提升，正在保存模型...")
            min_test_loss = test_loss
            torch.save(denoiser.state_dict(), DENOISER_MODEL_NAME)
            tqdm.write(f" 模型已保存到: {DENOISER_MODEL_NAME}")
        else:
            tqdm.write(f" 无改进（当前最优 Test Loss: {min_test_loss:.6f})")

    # 6️⃣ 训练结束
    print("\n🎯 训练完成！")
    print(f"最优测试误差: {min_test_loss:.6f}")
    print("✅ 所有过程已结束。")

