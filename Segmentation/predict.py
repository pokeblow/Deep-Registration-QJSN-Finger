import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from model.unet_model import ResNet34UnetPlus
from utils.rangegrow import Rangegrow
import matplotlib.image as mpimg
import time
import os
from torchvision import transforms
from PIL import Image
import scipy.ndimage
from JSNmeasurement.utils.rangegrow import *
import copy

def soomth(mask, kernel_size=5):
    x_list = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[j][i] == 0:
                x_list.append(j)
                break

    half = kernel_size // 2
    x_list_conv = []
    for k in range(half):
        x_list_conv.append(x_list[k])
    for k in range(half, len(x_list) - half):
        tmp = []
        tmp.append(x_list[k])
        for m in range(1, half + 1):
            tmp.append(x_list[k - m])
            tmp.append(x_list[k + m])
        x_list_conv.append(int(np.mean(tmp)))
    for k in range(len(x_list) - half, len(x_list)):
        x_list_conv.append(x_list[k])

    mask_new = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask_new[j][i] = 1
            if j == x_list_conv[i]:
                break
    return mask_new

def evaluateIndicator(mask, predic):
    mask[mask != 1] = 0
    mask_bool = (mask == 1)
    predic_bool = (predic == 1)
    and_mask = (predic_bool == 1) & (mask_bool == 1)
    or_mask = (predic_bool == 1) | (mask_bool == 1)
    sum1 = 0
    sum2 = 0

    h, w = predic_bool.shape
    for i in range(h):
        for j in range(w):
            if (and_mask[i][j] == True):
                sum1 = sum1 + 1
    for i in range(h):
        for j in range(w):
            if (or_mask[i][j] == True):
                sum2 = sum2 + 1

    IoU = sum1 / sum2

    TP = np.sum(((mask == 1) & (predic == 1)))
    FP = np.sum(((mask == 0) & (predic == 1)))
    TN = np.sum(((mask == 0) & (predic == 0)))
    FN = np.sum(((mask == 1) & (predic == 0)))

    print(TN)

    # SEM
    SEM = TP / (TP + FN)
    # SPC
    if (TN + FP) != 0:
        SPC = TN / (TN + FP)
    else:
        SPC = 0
    # DSC
    DSC = 2 * TP / (2 * TP + FP + FN)
    # ACC
    ACC = (TP + TN) / (TP + TN + FN + FP)

    return IoU, SEM, SPC, DSC, ACC


def measurement(mask):
    h, w = mask.shape
    upper_list = []
    for i in range(w):
        for j in range(h):
            if mask[j][i] == 1:
                upper_list.append([i, j])
                break

    lower_list = []
    for i in range(w):
        for j in range(h - 1, -1, -1):
            if mask[j][i] == 1:
                lower_list.append([i, j])
                break
    width = []
    for i in range(len(upper_list)):
        width.append(-(np.array(upper_list[i]) - np.array(lower_list[i]))[1])
    width = np.array(width)
    # print("Mean:", np.mean(width), "Max:", np.max(width), "Min:", np.min(width))

    return np.mean(width), np.max(width), np.min(width)


def predict(image_name, root_path, show=True):
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = ResNet34UnetPlus(num_channels=1, num_class=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model_seg_0309.pth', map_location=device), strict=False)
    # 测试模式
    net.eval()
    # 读取所有图片路径
    # tests_path = glob.glob('/Volumes/PortableSSD/BAT_Segmentation/data/PETCTTrainData/data/predict/*.jpg')
    image_name = image_name
    image_path = root_path + '/image/{}.bmp'.format(image_name)
    label_path = root_path + '/label/{}.bmp'.format(image_name)
    # 遍历所有图片

    # 读取图片
    image = np.array(Image.open(image_path))
    label = np.array(Image.open(label_path))

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    # 将数据转为图片
    image = transform(Image.fromarray(image))
    label = transform(Image.fromarray(label))

    image = np.array(image)
    background = copy.deepcopy(image[0])
    image = np.array([image])

    # image = image.transpose((0, 3, 2, 1))
    # 转为tensor
    img_tensor = torch.from_numpy(image)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    # 预测
    pred = net(img_tensor)
    # 提取结果
    pred = np.array(pred.data.cpu()[0])
    pred = pred[0]

    def normalization(data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    # 将数据转为图片
    image_range = 0.65

    pred = normalization(pred)

    pred[pred >= image_range] = 1
    pred[pred < image_range] = 0

    seeds = [Point(5, 250), Point(5, 5)]
    pred = regionGrow(pred, seeds, 1)

    pred = -pred
    pred[pred == -1] = 1

    seeds = [Point(250, 250), Point(250, 5)]
    pred = regionGrow(pred, seeds, 1)

    pred = pred - 1
    pred[pred == -1] = 1

    label = np.array(label[0])

    label = soomth(label, kernel_size=13)

    pred = 1 - pred
    label = 1 - label


    IoU, SEM, SPC, DSC, ACC = evaluateIndicator(label, pred)

    if show:
        # mpimg.imsave('utils/resultimage/{}_result.png'.format(image_name), pred)
        plt.figure(figsize=(9, 5))

        # kernel = np.ones((3, 3), np.float32) / 25
        # label = cv2.filter2D(label, -1, kernel)

        a = [i for i in range(label.shape[1])]
        b = [i for i in range(label.shape[0])]
        X, Y = np.meshgrid(a, b)

        a2 = [i for i in range(pred.shape[1])]
        b2 = [i for i in range(pred.shape[0])]
        X2, Y2 = np.meshgrid(a2, b2)

        plt.imshow(background, 'gray')

        plt.contour(X, Y, label, colors='w', linewidths=1)
        # plt.imshow(1-pred, cmap='GnBu', alpha=0.2)
        # plt.contour(X2, Y2, pred, colors='y', linewidths=1, alpha=0.5)
        plt.axis('off')
        file_name = '../Segmentation/Result/{}.png'.format(image_name)
        if IoU > 0.95:
            # plt.savefig(file_name, transparent=True)
            # plt.clf()
            print('IoU:{:.5f}, SEM:{:.5f}, SPC:{:.5f}, DSC:{:.5f}, ACC:{:.5f}'.format(IoU, SEM, SPC, DSC, ACC))

        plt.show()


    return IoU, SEM, SPC, DSC, ACC


if __name__ == "__main__":
    # image_list = ['010_0_0_010', '001_0_1_002', '001_1_1_003', '035_4_2_022', '061_3_2_000']
    # image_name = '010_0_0_010'
    root_path = '/Users/wanghaolin/PycharmProjects/Image_Registration/Data/finger_joint_selected_segment/test'
    dataset_file = os.listdir(root_path + '/' + 'image')
    for image_name in dataset_file:
        if image_name != '.DS_Store':
            image_name = image_name[:-4]
            if image_name[4:7] == '3_1':
                print(image_name)
                IoU, SEM, SPC, DSC, ACC = predict(image_name, root_path, show=True)

    # filepath = '/Users/wanghaolin/PycharmProjects/Image_Registration/Data/finger_joint_selected_segment/test/image'
    # root_path = '/Users/wanghaolin/PycharmProjects/Image_Registration/Data/finger_joint_selected_segment/test'
    # dataset_file = os.listdir(filepath)
    # num = len(dataset_file)
    # IoU_sum = 0
    # SEM_sum = 0
    # SPC_sum = 0
    # DSC_sum = 0
    # ACC_sum = 0
    # IP_lsit = [[], [], [], [], []]
    # PIP_lsit = [[], [], [], [], []]
    # MCP_lsit = [[], [], [], [], []]
    # for image_name in dataset_file:
    #     if image_name != '.DS_Store':
    #         image_name = image_name[:-4]
    #         if image_name[4:7] == '0_0':
    #             print(image_name, 'IP')
    #             IoU, SEM, SPC, DSC, ACC = predict(image_name, root_path, show=False)
    #             print('IoU:{:.5f}, SEM:{:.5f}, SPC:{:.5f}, DSC:{:.5f}, ACC:{:.5f}'.format(IoU, SEM, SPC, DSC, ACC))
    #             IP_lsit[0].append(IoU)
    #             IP_lsit[1].append(SEM)
    #             IP_lsit[2].append(SPC)
    #             IP_lsit[3].append(DSC)
    #             IP_lsit[4].append(ACC)
    #         else:
    #             if (image_name[4] == '0') & (image_name[6] == '1'):
    #                 print(image_name, 'MCP')
    #                 IoU, SEM, SPC, DSC, ACC = predict(image_name, root_path, show=False)
    #                 print('IoU:{:.5f}, SEM:{:.5f}, SPC:{:.5f}, DSC:{:.5f}, ACC:{:.5f}'.format(IoU, SEM, SPC, DSC, ACC))
    #                 MCP_lsit[0].append(IoU)
    #                 MCP_lsit[1].append(SEM)
    #                 MCP_lsit[2].append(SPC)
    #                 MCP_lsit[3].append(DSC)
    #                 MCP_lsit[4].append(ACC)
    #             if (image_name[4] != '0') & (image_name[6] == '2'):
    #                 print(image_name, 'MCP')
    #                 IoU, SEM, SPC, DSC, ACC = predict(image_name, root_path, show=False)
    #                 print('IoU:{:.5f}, SEM:{:.5f}, SPC:{:.5f}, DSC:{:.5f}, ACC:{:.5f}'.format(IoU, SEM, SPC, DSC, ACC))
    #                 MCP_lsit[0].append(IoU)
    #                 MCP_lsit[1].append(SEM)
    #                 MCP_lsit[2].append(SPC)
    #                 MCP_lsit[3].append(DSC)
    #                 MCP_lsit[4].append(ACC)
    #             if (image_name[4] != '0') & (image_name[6] == '1'):
    #                 print(image_name, 'PIP')
    #                 IoU, SEM, SPC, DSC, ACC = predict(image_name, root_path, show=False)
    #                 print('IoU:{:.5f}, SEM:{:.5f}, SPC:{:.5f}, DSC:{:.5f}, ACC:{:.5f}'.format(IoU, SEM, SPC, DSC, ACC))
    #                 PIP_lsit[0].append(IoU)
    #                 PIP_lsit[1].append(SEM)
    #                 PIP_lsit[2].append(SPC)
    #                 PIP_lsit[3].append(DSC)
    #                 PIP_lsit[4].append(ACC)
    #
    # print('IP, Average IoU:{:.5f}, SEM:{:.5f}, SPC:{:.5f}, DSC:{:.5f}, ACC:{:.5f}'.format(np.mean(IP_lsit[0]),
    #                                                                                       np.mean(IP_lsit[1]),
    #                                                                                       np.mean(IP_lsit[2]),
    #                                                                                       np.mean(IP_lsit[3]),
    #                                                                                       np.mean(IP_lsit[4])))
    # print('PIP, Average IoU:{:.5f}, SEM:{:.5f}, SPC:{:.5f}, DSC:{:.5f}, ACC:{:.5f}'.format(np.mean(PIP_lsit[0]),
    #                                                                                        np.mean(PIP_lsit[1]),
    #                                                                                        np.mean(PIP_lsit[2]),
    #                                                                                        np.mean(PIP_lsit[3]),
    #                                                                                        np.mean(PIP_lsit[4])))
    # print('MCP, Average IoU:{:.5f}, SEM:{:.5f}, SPC:{:.5f}, DSC:{:.5f}, ACC:{:.5f}'.format(np.mean(MCP_lsit[0]),
    #                                                                                        np.mean(MCP_lsit[1]),
    #                                                                                        np.mean(MCP_lsit[2]),
    #                                                                                        np.mean(MCP_lsit[3]),
    #                                                                                        np.mean(MCP_lsit[4])))
