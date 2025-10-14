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
from classification_config import *
from classification_data import ImageLabelDataSet
from classification_model import *
from classification_engine import *

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
    dataset = ImageLabelDataSet(IMG_PATH,LABELS_PATH, transform=transform)
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
    classifier =Classifier()
    classifier.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)
    print(f"✅ 模型加载完成，学习率: {LEARNING_RATE}")

    ## 初始化最小测试误差 以找到最优模型
min_test_loss = np.inf
best_acc = 0.0

# 4️⃣ 训练与测试循环
for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
    tqdm.write("\n" + "=" * 50)
    tqdm.write(f"🧩 Epoch [{epoch + 1}/{EPOCHS}] begin...\n")

    # ===== 训练阶段 =====
    train_loss = train_step(classifier, train_loader, loss, optimizer, device)
    tqdm.write(f"📈 Train Loss: {train_loss:.6f}")

    # ===== 测试阶段 =====
    test_loss, right_num = test_step(classifier, test_loader, loss, device)
    total_samples = len(test_dataset)
    accuracy = 100.0 * right_num / total_samples

    tqdm.write(f"🧪 Test Loss: {test_loss:.6f} | Accuracy: {accuracy:.2f}%")

    # ===== 保存模型逻辑 =====
    improved = False
    if test_loss < min_test_loss:
        tqdm.write(f"💾 Loss improved: {min_test_loss:.6f} → {test_loss:.6f}")
        min_test_loss = test_loss
        improved = True

    if accuracy > best_acc:
        tqdm.write(f"🏅 Accuracy improved: {best_acc:.2f}% → {accuracy:.2f}%")
        best_acc = accuracy
        improved = True

    if improved:
        torch.save(classifier.state_dict(), CLASSIFIER_MODEL_NAME)
        tqdm.write(f"✅ Model saved to: {CLASSIFIER_MODEL_NAME}")
    else:
        tqdm.write("⚠️ No improvement this epoch.")


    # 6️⃣ 训练结束
    print(f"best loss: {min_test_loss:.6f}")

    #测试
    # loaded_classifier = Convclassifier()
    # model_state_dict = torch.load(classifier_MODEL_NAME, map_location=device)
    # loaded_classifier.load_state_dict(model_state_dict)
    # loaded_classifier.to(device)
    #
    # # 测试
    # test(loaded_classifier, test_loader, device)
