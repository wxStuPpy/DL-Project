__all__ = ['train_step']

import torch


def train_step(denoiser, train_loader, loss, optimizer, device):
    # 执行一个轮次(epoch)的完整训练步骤
    # 设置为训练模式
    denoiser.train()
    # 累计损失值
    total_loss = 0
    # 遍历DataLoader
    for train_images, target_imgs in train_loader:
        train_images = train_images.to(device)
        target_imgs = target_imgs.to(device)

        outputs = denoiser(train_images)

        loss_value = loss(outputs, target_imgs)

        loss_value.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss_value.item()

    return total_loss / len(train_loader)


def test_step(denoiser, test_loader, loss, device):
    # 设置验证模式
    denoiser.eval()
    total_loss = 0
    with torch.no_grad():
        for test_images, target_imgs in test_loader:
            test_images = test_images.to(device)
            target_imgs = target_imgs.to(device)
            outputs = denoiser(test_images)
            loss_value = loss(outputs, target_imgs)
            total_loss += loss_value.item()
    return total_loss / len(test_loader)
