import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入自定义组件
from project.ITH.common import utils
from similarity_config import *
from similarity_data import ImageDataSet
from similarity_model import *
from similarity_engine import *

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

    full_loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE
    )

    # 3️⃣ 模型与优化器
    encoder = ConvEncoder()
    decoder = ConvDecoder()

    loss_fn = nn.MSELoss()

    # 联合编码器和解码器参数列表
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=LEARNING_RATE)
    print(f"✅ 模型加载完成，学习率: {LEARNING_RATE}")

    # 初始化最小测试误差 以找到最优模型
    min_test_loss = np.inf

    # # 4️⃣ 训练与测试循环
    # encoder.to(device)
    # decoder.to(device)
    # for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
    #     tqdm.write("\n" + "=" * 50)
    #     tqdm.write(f"🧩 Epoch [{epoch + 1}/{EPOCHS}] begin...\n")
    #
    #     # 训练一轮
    #     train_loss = train_step(encoder, decoder, train_loader, loss_fn, optimizer, device)
    #     tqdm.write(f"📈 Epoch [{epoch + 1}/{EPOCHS}]  - Train Loss: {train_loss:.6f}")
    #
    #     # 测试一轮
    #     test_loss = test_step(encoder, decoder, test_loader, loss_fn, device)
    #     tqdm.write(f"🧪 Epoch [{epoch + 1}/{EPOCHS}]  - Test Loss: {test_loss:.6f}")
    #
    #     # 判断并保存模型
    #     if test_loss < min_test_loss:
    #         tqdm.write(f" descend: {min_test_loss:.6f} → {test_loss:.6f}")
    #         tqdm.write(" saving model...")
    #         min_test_loss = test_loss
    #         torch.save(encoder.state_dict(), ENCODE_MODEL_NAME)
    #         torch.save(decoder.state_dict(), DECODE_MODEL_NAME)
    #     else:
    #         tqdm.write('not saving model')
    #
    # # 6️⃣ 训练结束
    # print(f"best loss: {min_test_loss:.6f}")

    # 生成图形嵌入表达
    encoder_state_dict = torch.load(ENCODE_MODEL_NAME,map_location=device)
    encoder.load_state_dict(encoder_state_dict)

    embeddings = create_embedding(encoder, full_loader, device)

    # 转化为numpy保存
    vec_embeddings = embeddings.detach().numpy().reshape(embeddings.shape[0], -1)

    # 保存到文件
    np.save(EMBEDDING_NAME, vec_embeddings)

    print(f'embeddings shape: {vec_embeddings.shape}')
    print(f'vec embeddings shape: {vec_embeddings.shape}')
