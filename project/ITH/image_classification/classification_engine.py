__all__ = ['train_step', 'test_step']

import torch


def train_step(classifier, train_loader, loss_fn, optimizer, device):
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = classifier(images)
        loss_value = loss_fn(outputs, labels)

        loss_value.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss_value.item()

    this_loss = total_loss / len(train_loader)

    return this_loss


def test_step(classifier, test_loader, loss_fn, device):
    total_loss = 0
    right_num = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            total_loss += loss_fn(outputs, labels).item()

            pred = outputs.argmax(dim=1)
            right_num += pred.eq(labels).sum().item()

        return total_loss / len(test_loader), right_num
