import argparse
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import json
import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import clip
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import scipy.sparse as sp
import torch.optim as optim
import random
from utils import progress_bar


# 导入train.py中定义的模型和函数

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 关键！确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭基准优化，保证可重复性

set_seed() # 调用函数设置种子


# 1. 模型定义
class MultiTaskModel(nn.Module):
    def __init__(self, num_aux1_classes, num_aux2_classes):
        super(MultiTaskModel, self).__init__()
        model = timm.create_model('mobilenetv2_100', pretrained=False)
        # 加载本地预训练权重
        checkpoint_path = './checkpoint/new_mv2-2.pth'
        print(f"Loading MobileNetV2 weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cuda')

        model.load_state_dict(state_dict)
        print("MobileNetV2 weights loaded successfully")

        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

        # 辅助任务1头
        self.auxiliary_task1_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Linear(512, num_aux1_classes)
        )

        # 辅助任务2头
        self.auxiliary_task2_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Linear(512, num_aux2_classes)
        )

    def forward(self, x):
        # 使用特征提取器提取特征
        features = self.feature_extractor(x)
        # 将特征展平
        # features = features.view(features.size(0), -1)
        # main_task_output = self.main_task_head(features)
        aux1_task_output = self.auxiliary_task1_head(features)
        aux2_task_output = self.auxiliary_task2_head(features)
        enc_layers = list(self.auxiliary_task1_head.children())
        enc_1 = nn.Sequential(*enc_layers[:1])  # input -> 512
        enc_layers = list(self.auxiliary_task2_head.children())
        enc_2 = nn.Sequential(*enc_layers[:1])  # input -> 512
        return aux1_task_output, aux2_task_output, enc_1(features), enc_2(features)


def Scene_model(img):
    # th architecture to use """场景特征提取器（使用Places365预训练的ResNet18）"""
    arch = 'resnet18'

    # 加载预训练的权重
    model_file = './checkpoint/resnet18_places365.pth.tar'

    # 加载模型
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 特征提取
    model.to(device)
    logit = model(img)
    # print(logit.shape)  # torch.Size([b, 365]) # 填充到512维（适配GCN输入）
    target_shape = (logit.shape[0], 512)
    padding_size = target_shape[1] - logit.shape[1]
    padding_tensor = torch.zeros(logit.shape[0], padding_size)
    padding_tensor = padding_tensor.to(device)
    result = torch.cat((logit, padding_tensor), dim=1)
    result = result.to(device)
    return result

#######################   GCN
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum == 0] = 0.0000001
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.00000001
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class GraphConvolution(nn.Module):
    """图卷积层"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # 权重
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  # 偏置
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        # Xavier初始化
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)  # XW
        output = torch.matmul(adj, support)  # AXW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    """图卷积网络（情感分类器）"""

    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)  # GCN层1
        self.gc2 = GraphConvolution(nhid, nhid * 2)  # GCN层2
        self.dropout = dropout
        # 全连接分类器
        self.fc1 = nn.Linear(nhid * 2 * 4, 256)  # 4个节点
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)  # 8种情感

    def forward(self, x, adj):
        # print('1, ', x.shape)       #torch.Size([batchsize, 6, 256])
        x = F.relu(self.gc1(x, adj))
        # print('1, ', x.shape)       #torch.Size([batchsize, 6, 1024])
        x = F.dropout(x, self.dropout, training=self.training)
        # print('2, ', x.shape)       #torch.Size([batchsize, 6, 1024])
        x = F.relu(self.gc2(x, adj))
        # print('3, ', x.shape)       #torch.Size([batchsize, 6, 2048])
        x = x.view(x.shape[0], 1, -1)
        # print('4, ', x.shape)       #torch.Size([batchsize, 1, 12288])
        if x.ndim == 2:  # 处理单样本情况
            x = x.view(1, -1)
        # print('5, ', x.shape)       #torch.Size([batchsize, 1, 12288])
        # 全连接分类
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print('6, ', x.shape)       #torch.Size([batchsize, 1, 8])
        x = x.reshape((x.shape[0], -1))
        # print('7, ', x.shape)       #torch.Size([batchsize, 8])
        return x


# 情感类别映射字典
emotion_dict = {
    'amusement': 0,
    'awe': 1,
    'contentment': 2,
    'excitement': 3,
    'anger': 4,
    'disgust': 5,
    'fear': 6,
    'sadness': 7,
}


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, train=True):
        self.data_path = data_path
        self.transform = transform
        self.train = train
        self.filenames = []
        self.targets = []

        # with open(data_path, 'r') as f:
        #     lines = f.readlines()
        with open(data_path, 'r') as json_file:
            data = json.load(json_file)

        for sub_list in data:
            if len(sub_list) == 3:
                emotion = sub_list[0]
                image_path = sub_list[1]
                json_path = sub_list[2]
            else:
                print("Invalid sub-list format")

            image_dir = r'D:\DataSets'
            filename = os.path.join(image_dir, image_path)
            target = emotion_dict[emotion]
            self.filenames.append(filename)
            self.targets.append(target)
            # self.filenames = [os.path.join(image_dir, filename) for filename in self.filenames]
        print(f"Loaded {len(self.filenames)} images. Class distribution:")
        print(np.bincount(self.targets))

    def __len__(self):

        return len(self.filenames)

    def __getitem__(self, index):

        # and return a tuple (data, label)
        filename, target = self.filenames[index], self.targets[index]

        img = Image.open(filename)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(target)




# def test(model_GCN, Attribute_model, clip_model, test_loader, device, adj):
#     """
#     测试模型
#     """
#     all_preds = []
#     all_targets = []

#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             inputs, targets = inputs.to(device), targets.to(device)

#             # 提取各种特征
#             Scene_f = Scene_model(inputs)
#             clip_f = clip_model.encode_image(inputs)
#             Brightness_x, Colorfulness_x, Brightness_f, Colorfulness_f = Attribute_model(inputs)

#             # 构建图节点特征
#             temp = torch.zeros(Brightness_f.shape[0], 4, Brightness_f.shape[1])
#             for num in range(Brightness_f.shape[0]):
#                 temp[num::] = torch.stack((Scene_f[num, :], Brightness_f[num, :],
#                                            Colorfulness_f[num, :], clip_f[num, :]), 0)

#             temp = temp.to(device)

#             # 通过GCN进行预测
#             outputs = model_GCN(temp, adj)
#             _, predicted = outputs.max(1)

#             # 保存预测结果和真实标签
#             all_preds.extend(predicted.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())

#             print(f'Processed batch {batch_idx + 1}/{len(test_loader)}')

#     return np.array(all_preds), np.array(all_targets)


# def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
#     """
#     绘制并保存混淆矩阵
#     """
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()


def main(lr=0.001,gpus=0):
    # 设置主设备
    global device
    
    # 检查请求的 GPU 是否可用
    requested_gpu = gpus
    available_gpus = torch.cuda.device_count()
    
    print(f"Requested GPU: {requested_gpu}")
    print(f"Available GPUs: {available_gpus}")
    
    # 确保请求的 GPU 在可用范围内
    if requested_gpu >= available_gpus:
        print(f"Warning: Requested GPU {requested_gpu} is not available. Using GPU 0 instead.")
        device = torch.device("cuda:0")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        # 尝试访问请求的 GPU
        try:
            # 创建一个测试张量来检查 GPU 是否可用
            test_tensor = torch.tensor([1.0]).to(f"cuda:{requested_gpu}")
            device = torch.device(f"cuda:{requested_gpu}")
            print(f"Using GPU {requested_gpu}: {torch.cuda.get_device_name(requested_gpu)}")
        except Exception as e:
            print(f"Error accessing GPU {requested_gpu}: {e}")
            print("Falling back to GPU 0")
            device = torch.device("cuda:0")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(f"Using device: {device}")

    #####################  GCN
    model_GCN = GCN(nfeat=512, nhid=1024, dropout=0.5).to(device)
    # 加载检查点
    checkpoint_path = './checkpoint/Emoset_3.0/test.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_GCN.load_state_dict(checkpoint['net'])
    global best_acc  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    best_acc = 0

    # Model
    print('==> Building model..')

    net = model_GCN



    Attribute_model = MultiTaskModel(11, 11)
    Attribute_model = Attribute_model.to(device)

    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model.to(device)



    criterion = nn.CrossEntropyLoss()
   
    edges_unordered = np.genfromtxt("./cora.cites", dtype=np.int32)  # 读入边的信息
    adj = np.zeros((4, 4))
    for [q, p] in edges_unordered:
        adj[q - 1, p - 1] = 1
    adj = torch.from_numpy(adj)
    adj = normalize(adj)
    adj = torch.from_numpy(adj)
    adj = adj.clone().float()
    adj = adj.to(device)


    # 准备测试数据
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    from torch.utils.data import Subset  # ✅ 确保导入

    testset = EmotionDataset(data_path='Datasets/EmoSet-118k/test.json', train=False,
                             transform=transform_test)

    # ✅ 只取前2000张
    subset_indices = list(range(2000))
    test_subset = Subset(testset, subset_indices)

    print(f"Using {len(test_subset)} test samples (first 200).")

    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=100, shuffle=False, num_workers=2)
    epoch_acc = checkpoint['acc']
    best_acc=checkpoint['best_acc']
    # 进行测试
    print("Starting testing...")
    print(f'Resumed from epoch{checkpoint["epoch"]},epoch_acc:{epoch_acc}%,best_acc:{best_acc}%')
    # all_preds, all_targets = test(net, Attribute_model, clip_model, test_loader, device, adj)
    net.eval()
    clip_model.eval()
    Attribute_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    # print("Keys in the eckpoint:", net.state_dict())
    # print("Optimizer state keys:", optimizer.state_dict())
    # print("Scheduler state keys:", scheduler.state_dict())
    # print("Attribute_model state keys:", Attribute_model.state_dict())
    # print("clip_model state keys:", clip_model.state_dict())
    # exit()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            Scene_f = Scene_model(inputs)
            # clip_f = clip_model.encode_image(inputs)
            clip_f = clip_model.encode_image(inputs)
            Brightness_x, Colorfulness_x, Brightness_f, Colorfulness_f = Attribute_model(inputs)

            temp = torch.zeros(Brightness_f.shape[0], 4, Brightness_f.shape[1])
            for num in range(Brightness_f.shape[0]):
                temp[num::] = torch.stack((Scene_f[num, :], Brightness_f[num, :],
                                           Colorfulness_f[num, :], clip_f[num, :]), 0)

            temp = temp.to(device)
            outputs = net(temp, adj)

            # ---------------------------------------------------------------------
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    print(acc)


if __name__ == '__main__':
    import argparse
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Emotion Classification Training on FI Dataset')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    # parser.add_argument('--fold', type=int, default=0, help='Fold number for K-fold cross validation')
    parser.add_argument('--gpus', type=int, default=0, help='GPU to use (e.g., "0" or "1")')
    args = parser.parse_args()
    
    # 设置使用的GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPU: {args.gpus}")
    # 传递所有参数给 main 函数
    main(lr=args.lr,gpus=args.gpus)