from model.unet_model import ResNet34UnetPlus
from utils.dataset import ISBI_Loader
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import torch.nn as nn
import torch
import time
from torchsummary import summary
from torchvision import transforms
import os


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001, rise=160):
    # 加载训练集
    transform = transforms.Compose([transforms.Resize((rise, rise)),
                                    transforms.ToTensor()])
    isbi_dataset = ISBI_Loader(data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    print('Data Loaning finished')
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR()
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    Loss_list = []
    loss_list_epoch = []
    for epoch in range(epochs):
        print('Epoch:', epoch + 1, '/', epochs)
        net.train()
        step = 0
        for image, label in train_loader:
            # 时间START
            start = time.time()
            step = step + 1
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            Loss_list.append(loss.item())
            loss_list_epoch.append(loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model_seg.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
            # 时间END
            end = time.time()
            print('Epoch:', epoch + 1, 'Step:', step, '/', len(train_loader),
                  'Training Loss:', loss.item(),
                  'Time spending:', end - start)

        loss_var = np.var(loss_list_epoch)
        print('Epoch:', epoch + 1, 'Loss_var:', loss_var)
        loss_list_epoch = []

    # x2 = range(0, len(Loss_list))
    # y2 = Loss_list
    # fig = plt.figure()
    # plt.plot(x2, y2, '.-')
    # plt.xlabel('Step')
    # plt.ylabel('Loss')
    # plt.show()
    # fig.savefig("loss.jpg")
    # f = open('loss.txt', mode='w')
    # for i in Loss_list:
    #     f.write(str(i)+'\n')
    # f.close()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training Device:', device)
    # 加载网络，图片单通道1，分类为1。
    net = ResNet34UnetPlus(num_channels=1, num_class=1)
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "Data/finger_joint_selected_segment"
    start = time.time()
    train_net(net, device, data_path, epochs=150, batch_size=10, lr=0.00001, rise=256)
    end = time.time()
    print('Training Finished, Time Spends:', end - start)
