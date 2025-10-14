import torch

def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    encoder.train()
    decoder.train()
    total_loss = 0
    for train_imgs, target_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        target_imgs = target_imgs.to(device)

        optimizer.zero_grad()

        en_outputs = encoder(train_imgs)
        de_outputs = decoder(en_outputs)

        loss_val = loss_fn(de_outputs, target_imgs)
        loss_val.backward()
        optimizer.step()
        total_loss += loss_val.item()

    return total_loss / len(train_loader)

def test_step(encoder, decoder, test_loader, loss_fn, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for test_imgs, target_imgs in test_loader:
            test_imgs = test_imgs.to(device)
            target_imgs = target_imgs.to(device)

            en_outputs = encoder(test_imgs)
            de_outputs = decoder(en_outputs)
            loss_val = loss_fn(de_outputs, target_imgs)
            total_loss += loss_val.item()

    return total_loss / len(test_loader)

def create_embedding(encoder, full_loader, device):
    encoder.eval()
    #定义嵌入张量,初始为空
    embeddings = torch.empty(0)
    with torch.no_grad():
        for train_imgs, _ in full_loader:
            train_imgs = train_imgs.to(device)
            #前向传播
            encode_img = encoder(train_imgs).cpu()
            #将这批次结果 拼接到嵌入张量中
            embeddings = torch.cat((embeddings, encode_img), 0)
    #返回嵌入的张量
    return embeddings

