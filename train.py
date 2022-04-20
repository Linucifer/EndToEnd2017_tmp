import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from steg_net import StegNet

# 定义参数
configs = {
    'train_rate': 0.8,  # 训练数据占数据总量的比例
    'host_channels': 3,
    'guest_channels': 1,
    'img_width': 32,
    'img_height': 32,
    'epoch_num': 50,
    'train_batch_size': 256,
    'val_batch_size': 64,
    'encoder_weight': 1,
    'decoder_weight': 1,
    'model_path': '/content/drive/MyDrive/MyModels/End_to_end_Stegnography_2017',
    'learning_rate': 1e-4
}

# 下载训练图像数据
cifar10_data = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True).data

# 定义图像数据转换
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# 建立训练数据集
train_dataset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=False, transform=transform)
train_dataset.data = cifar10_data[0:int(50000 * configs['train_rate'])]
train_dataset_loader = DataLoader(train_dataset,
                                  batch_size=configs['train_batch_size'],
                                  shuffle=True)
# 建立验证数据集
val_dataset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=False, transform=transform)
val_dataset.data = cifar10_data[int(50000 * configs['train_rate']):]
val_dataset_loader = DataLoader(val_dataset,
                                batch_size=configs['val_batch_size'],
                                shuffle=True)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running device is {device.type}")

# 使用默认参数构建模型
model = StegNet()
model.to(device)
# model.load_model(configs['model_path'], file_name=f"steg_net"
#                                                   f"_host{configs['host_channels']}"
#                                                   f"_guest{configs['guest_channels']}"
#                                                   f"_epochNum{configs['epoch_num']}"
#                                                   f"_batchSize{configs['train_batch_size']}"
#                                                   f"_lr{configs['learning_rate']}"
#                                                   f"_encoderWeight{configs['encoder_weight']}"
#                                                   f"_decoderWeight{configs['decoder_weight']}.pth")

# 定义损失函数
criterion = nn.MSELoss()
metric = nn.L1Loss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])
# 最小的图像损失
min_val_loss = +np.inf

for epoch in range(configs['epoch_num']):
    train_loss = 0
    model.train()

    for batch in tqdm(train_dataset_loader, desc=f"Train Epoch: {epoch}"):
        # 使用数据集的图像部分作为本模型的训练数据
        train_data, _ = [x.to(device) for x in batch]
        # 将一半数据作为host 另一半数据作为guest
        host_img = train_data[0:int(train_data.shape[0] / 2)]
        guest_img = train_data[int(train_data.shape[0] / 2):]
        # 使用guest的一个通道作为载密图像
        guest_img = torch.tensor([x[0].tolist() for x in guest_img]).unsqueeze(1).to(device)

        optimizer.zero_grad()
        encoder_out, decoder_out = model(host_img, guest_img)

        # 计算均方差损失
        encoder_loss = criterion(encoder_out.view(-1, configs['img_width'] * configs['img_height']),
                                 host_img.view(-1, configs['img_width'] * configs['img_height']))
        decoder_loss = criterion(decoder_out.view(-1, configs['img_width'] * configs['img_height']),
                                 guest_img.view(-1, configs['img_width'] * configs['img_height']))
        loss = configs['encoder_weight'] * encoder_loss + configs['decoder_weight'] * decoder_loss
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()

    else:
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataset_loader, desc=f"Val Epoch {epoch}"):
                # 使用数据集的图像部分作为本模型的训练数据
                val_data, _ = [x.to(device) for x in batch]
                # 将一半数据作为host 另一半数据作为guest
                host_img = val_data[0:int(val_data.shape[0] / 2)]
                guest_img = val_data[int(val_data.shape[0] / 2):]
                # 使用guest的一个通道作为载密图像
                guest_img = torch.tensor([x[0].tolist() for x in guest_img]).unsqueeze(1).to(device)

                encoder_out, decoder_out = model(host_img, guest_img)

                # 计算均方差损失
                encoder_loss = metric(encoder_out.view(-1, configs['img_width'] * configs['img_height']),
                                      host_img.view(-1, configs['img_width'] * configs['img_height']))
                decoder_loss = metric(decoder_out.view(-1, configs['img_width'] * configs['img_height']),
                                      guest_img.view(-1, configs['img_width'] * configs['img_height']))
                loss = encoder_loss + decoder_loss
                val_loss = val_loss + loss.item()

    train_loss = train_loss / len(train_dataset_loader)
    val_loss = val_loss / len(val_dataset_loader)
    print(f"Epoch {epoch} train_loss: {train_loss}")
    print(f"Epoch {epoch} val_loss: {val_loss}")
    if val_loss <= min_val_loss:
        print(f"Validation loss decreased {min_val_loss} --> {val_loss}")
        min_val_loss = val_loss
        model.save_model(configs['model_path'], file_name=f"steg_net"
                                                          f"_host{configs['host_channels']}"
                                                          f"_guest{configs['guest_channels']}"
                                                          f"_epochNum{configs['epoch_num']}"
                                                          f"_batchSize{int(configs['train_batch_size']/2)}"
                                                          f"_lr{configs['learning_rate']}"
                                                          f"_encoderWeight{configs['encoder_weight']}"
                                                          f"_decoderWeight{configs['decoder_weight']}.pth")
