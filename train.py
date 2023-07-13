import torch
from torch import optim
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms
from torch.autograd import gradcheck

import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

from train_mask.dataset import Data_Loader
import lossfuction as lf
from model.model_VGG import Registration_VGG
from model.model_CNN import Registration_CNN
# from model.model_ResCNN import Registration_ResCNN
from model.model_VGGPlus import Registration_VGGPlus
from model.model_ResCNN_mask import Registration_ResCNN
from model.model_ResCNN_mask_2 import Registration_ResCNN_2
from model.model_ResCNN_mask_3 import Registration_ResCNN_3


def train(net, pretrain_path, device, data_path, epochs=40, batch_size=1, lr=0.00001, resize=160):
    train_Loss_list_step = []
    train_loss_list_epoch = []
    train_loss_var_list = []
    train_loss_mean_list = []
    lr_out = lr

    # dataset loading
    print('Training Device:', device)
    transform = transforms.Compose([transforms.Resize((resize, resize)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0, 1)
                                    ])
    image_dataset = Data_Loader(data_path, transform)

    train_loader = torch.utils.data.DataLoader(dataset=image_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    print('Data Loading Finished')
    # pertrain parameters loading
    # net.load_state_dict(torch.load(pretrain_path, map_location=torch.device('cpu')))
    # optimizer definition
    optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10,
                                                           verbose=False,
                                                           threshold=0.00001, threshold_mode='rel', cooldown=0,
                                                           min_lr=0,
                                                           eps=1e-20
                                                           )
    # loss definition
    criterion = nn.MSELoss()

    criterion_upper = nn.MSELoss()
    criterion_lower = nn.MSELoss()
    best_loss = float('inf')

    training_image_show = []

    # train epoch
    for epoch in range(epochs):
        print('Epoch:', epoch + 1, '/', epochs, 'lr:', lr_out)
        lr_out = optimizer.state_dict()['param_groups'][0]['lr']
        net.train()
        step = 0
        for moving, fixed, moving_seg, fixed_seg in train_loader:
            # data loading
            moving = moving.to(device=device, dtype=torch.float32)
            fixed = fixed.to(device=device, dtype=torch.float32)
            moving_seg = moving_seg.to(device=device, dtype=torch.float32)
            fixed_seg = fixed_seg.to(device=device, dtype=torch.float32)

            # forward
            optimizer.zero_grad()

            moving_reg, fixed_crop, output = net(moving, fixed, moving_seg, fixed_seg)

            if step % 5 == 0:
                training_image_show.append(
                    [moving_reg.cpu().detach().numpy()[0][0], output.cpu().detach().numpy()[0]])

            # loss
            loss_upper = criterion_upper(moving_reg[:, 0], fixed_crop[:, 0])
            loss_lower = criterion_lower(moving_reg[:, 1], fixed_crop[:, 1])

            loss = loss_upper + loss_lower

            loss.backward()
            optimizer.step()

            step = step + 1
            train_Loss_list_step.append(loss.item())
            train_loss_list_epoch.append(loss.item())
            print('Epoch:', epoch + 1, 'Step:', step, '/', len(train_loader),
                  'Training Loss: {:.12f}'.format(loss.item()),
                  'Upper Loss: {:.12f}'.format(loss_upper.item()),
                  'Lower Loss: {:.12f}'.format(loss_lower.item()),
                  'LR:', optimizer.state_dict()['param_groups'][0]['lr'],
                  'Output_upper: S({:.5f}) R({:.5f}) T({:.5f}, {:.5f})'.format(output.cpu().detach().numpy()[0][0][0],
                                                                         output.cpu().detach().numpy()[0][0][1],
                                                                         output.cpu().detach().numpy()[0][0][2],
                                                                         output.cpu().detach().numpy()[0][0][3]),
                  'Output_lower: S({:.5f}) R({:.5f}) T({:.5f}, {:.5f})'.format(output.cpu().detach().numpy()[0][1][0],
                                                                         output.cpu().detach().numpy()[0][1][1],
                                                                         output.cpu().detach().numpy()[0][1][2],
                                                                         output.cpu().detach().numpy()[0][1][3]))

        # 评价epoch
        train_loss_var = np.var(train_loss_list_epoch)
        train_loss_mean = np.mean(train_loss_list_epoch)
        train_loss_var_list.append(train_loss_var)
        train_loss_mean_list.append(train_loss_mean)
        # 保存最优模型
        if train_loss_mean < best_loss:
            best_loss = train_loss_mean
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module, 'model/parameters/best_Dmodel_1.pth')
            else:
                torch.save(net, 'model/parameters/best_model_1.pth')

        # lr refresh
        scheduler.step(train_loss_mean)

        print('Epoch:', epoch + 1, 'Train_Loss_var:', train_loss_var, 'Train_Loss_mean:', train_loss_mean)
        train_loss_list_epoch = []
        # show image
        plt.figure(figsize=(12.0, 6.0))
        image_show_len = len(training_image_show)
        for i in range(image_show_len):
            plt.subplot(2, image_show_len // 2, i + 1)
            plt.imshow(training_image_show[i][0][0])
            # plt.imshow(training_image_show[i][0][1], cmap='gray', alpha=0.5)
            #             plt.title('S({:.5f}) R({:.5f}) T({:.5f}, {:.5f})'.format(training_image_show[i][1][0],
            #                                                                      training_image_show[i][1][1],
            #                                                                      training_image_show[i][1][2],
            #                                                                      training_image_show[i][1][3]))
            plt.axis('off')
        plt.show()

        plt.figure(figsize=(12.0, 6.0))
        image_show_len = len(training_image_show)
        for i in range(image_show_len):
            plt.subplot(2, image_show_len // 2, i + 1)
            plt.imshow(training_image_show[i][0][1])
            #             plt.title('S({:.5f}) R({:.5f}) T({:.5f}, {:.5f})'.format(training_image_show[i][1][0],
            #                                                                      training_image_show[i][1][1],
            #                                                                      training_image_show[i][1][2],
            #                                                                      training_image_show[i][1][3]))
            plt.axis('off')
        plt.show()
        training_image_show = None

    x2 = range(0, len(train_loss_mean_list))
    y2 = train_loss_mean_list
    fig = plt.figure()
    plt.plot(x2, y2, '.-')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()
    fig.savefig("TL_loss.jpg")
    f = open('TL_loss.txt', mode='w')
    for i in train_loss_mean_list:
        f.write(str(i) + '\n')
    f.close()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Registration_ResCNN_3()
    net.to(device=device)
    data_path = "../Data/phantom_dataset"
    pretrain_path = 'pretain model path'
    start = time.time()
    train(net=net, pretrain_path=pretrain_path, device=device, data_path=data_path, epochs=30, batch_size=10, lr=0.001,
          resize=224)
    end = time.time()
    print('Training Finished, Time Spends:', end - start)
