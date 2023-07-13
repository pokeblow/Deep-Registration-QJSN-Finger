import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import time
from torchsummary import summary
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2
import random
import utils.library as lb
import utils.retinex as retinex
import os
import copy
from openpyxl import Workbook
from openpyxl import load_workbook

from model.model_VGG import Registration_VGG
from model.model_CNN import Registration_CNN
from model.model_ResCNN_mask import Registration_ResCNN
# from model.model_ResCNN_mask_2 import Registration_ResCNN_2
from model.model_ResCNN_mask_3 import Registration_ResCNN_3
from model.model_VGGPlus import Registration_VGGPlus


def inputimage(image):
    segmentation = lb.separation(lb.gullyDetect(image))[0]

    segmentation[segmentation != 0] = 1

    range = 4

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = retinex.MSRCP(
        img=image,
        sigma_list=[15, 80, 200],
        low_clip=0.01,
        high_clip=0.99,
    )
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.medianBlur(image, 1)

    h, w = image.shape
    segmentation = segmentation[range: h - range, range: w - range]
    image = image[range: h - range, range: w - range]
    # plus = 25
    # segmentation = segmentation[range: h - range, range + plus: w - range - plus]
    # image = image[range: h - range, range + plus: w - range - plus]

    # return image, segmentation
    return image, segmentation


def inputimage_path(image_path):
    image = Image.open(image_path)

    image = np.array(image)

    segmentation = lb.separation(lb.gullyDetect(image))[0]

    segmentation[segmentation != 0] = 1

    h, w = image.shape

    range_out = 4
    image = image[range_out: h - range_out, range_out: w - range_out]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = retinex.MSRCP(
        img=image,
        sigma_list=[15, 80, 200],
        low_clip=0.01,
        high_clip=0.99,
    )
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.medianBlur(image, 1)

    segmentation = segmentation[range_out: h - range_out, range_out: w - range_out]

    # plus = 25
    # segmentation = segmentation[range: h - range, range + plus: w - range - plus]
    # image = image[range: h - range, range + plus: w - range - plus]

    return image, segmentation


def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range


def normalization_both(image_1, image_2):
    range = np.max(image_1) - np.min(image_1)
    image_1 = (image_1 - np.min(image_1)) / range
    image_2 = (image_2 - np.min(image_2)) / range
    return image_1, image_2, np.max(image_1)


def predict(moving, fixed, moving_seg, fixed_seg, length, filename, save, output_path, joint):
    plt.imshow(moving)
    plt.show()
    print(moving.shape)
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = Registration_ResCNN_3()
    # 加载模型参数
    # best_model_0127 best_model_0217_P
    net.load_state_dict(torch.load('../model/parameters/best_model_0301.pth', map_location='cpu'))
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 测试模式
    net.eval()
    criterion = nn.MSELoss(size_average=1, reduce=0)
    criterion2 = nn.MSELoss()

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    # 将数据转为图片
    moving = transform(Image.fromarray(moving))
    fixed = transform(Image.fromarray(fixed))
    moving_seg = transform(Image.fromarray(moving_seg))
    fixed_seg = transform(Image.fromarray(fixed_seg))

    moving = np.array(moving)
    moving = np.array([moving])
    fixed = np.array(fixed)
    fixed = np.array([fixed])
    moving_seg = np.array(moving_seg)
    movin_mask = copy.deepcopy(moving_seg[0])
    moving_seg = np.array([moving_seg])
    fixed_seg = np.array(fixed_seg)
    fixed_seg = np.array([fixed_seg])

    # 转为tensor
    moving_tensor = torch.from_numpy(moving)
    moving_tensor = moving_tensor.to(device=device, dtype=torch.float32)

    fixed_tensor = torch.from_numpy(fixed)
    fixed_tensor = fixed_tensor.to(device=device, dtype=torch.float32)

    moving_seg_tensor = torch.from_numpy(moving_seg)
    moving_seg_tensor = moving_seg_tensor.to(device=device, dtype=torch.float32)

    fixed_seg_tensor = torch.from_numpy(fixed_seg)
    fixed_seg_tensor = fixed_seg_tensor.to(device=device, dtype=torch.float32)

    moving_crop_image = moving_tensor.detach().numpy()[0][0]
    range_crop = 5
    cols, rows = moving_crop_image.shape
    moving_crop_image = moving_crop_image[range_crop: cols - range_crop, range_crop: rows - range_crop]
    movin_mask = movin_mask[range_crop: cols - range_crop, range_crop: rows - range_crop]

    fixed_org_image = fixed_tensor.detach().numpy()[0][0]
    cols, rows = fixed_org_image.shape
    fixed_org_image = fixed_org_image[range_crop: cols - range_crop, range_crop: rows - range_crop]

    # loss
    moving_tensor_pre = moving_tensor[:, :, range_crop: 224 - range_crop, range_crop: 224 - range_crop]
    fixed_tensor_pre = fixed_tensor[:, :, range_crop: 224 - range_crop, range_crop: 224 - range_crop]

    loss = criterion(moving_tensor_pre, fixed_tensor_pre)
    loss_val = criterion2(moving_tensor_pre, fixed_tensor_pre)
    loss_map_org = loss.detach().numpy()[0][0]

    # 预测
    moving_reg, fixed_crop, output = net(moving_tensor, fixed_tensor, moving_seg_tensor, fixed_seg_tensor)
    result = output.detach().numpy()[0]

    upper_result = result[0]
    lower_result = result[1]

    # print('Org_loss:', loss_val)
    #
    # print('Output_upper: S({:.5f}) R({:.5f}) T({:.5f}, {:.5f})'.format(upper_result[0], upper_result[1],
    #                                                                    upper_result[2], upper_result[3]))
    # print('Output_lower: S({:.5f}) R({:.5f}) T({:.5f}, {:.5f})'.format(lower_result[0], lower_result[1],
    #                                                                    lower_result[2], lower_result[3]))
    # print('Difference: S({:.5f}) R({:.5f}) T({:.5f}, {:.5f})'.format((upper_result - lower_result)[0],
    #                                                                  (upper_result - lower_result)[1],
    #                                                                  (upper_result - lower_result)[2],
    #                                                                  (upper_result - lower_result)[3] * (length / 224)))

    moving_reg_image = moving_reg.detach().numpy()[0]
    fixed_crop_image = fixed_crop.detach().numpy()[0]

    loss_upper = criterion(moving_reg[:, 0], fixed_crop[:, 0])
    loss_upper_map = loss_upper.detach().numpy()[0]
    loss_lower = criterion(moving_reg[:, 1], fixed_crop[:, 1])
    loss_lower_map = loss_lower.detach().numpy()[0]

    loss_result = 0.5 * criterion2(moving_reg[:, 0], fixed_crop[:, 0]) + 0.5 * criterion2(moving_reg[:, 1],
                                                                                          fixed_crop[:, 1])
    # Moving
    ## Upper
    moving_mask_upper = copy.deepcopy(moving_reg_image[0])
    moving_mask_upper[moving_mask_upper != 0] = 1
    ## Lower
    moving_mask_lower = copy.deepcopy(moving_reg_image[1])
    moving_mask_lower[moving_mask_lower != 0] = 1

    moving_mask_lower = moving_mask_lower - 1
    moving_mask_lower[moving_mask_lower == -1] = 1

    moving_mask = np.array((moving_mask_lower == 1) & (moving_mask_upper == 1))
    # moving_mask = np.array((moving_mask_upper == 1))

    tmp_mask = moving_mask

    moving_reg_image[0][tmp_mask == 0] = 0
    loss_upper_map[tmp_mask == 0] = 0

    moving_reg_image[1][tmp_mask == 1] = 0
    loss_lower_map[tmp_mask == 1] = 0

    moving_result = moving_reg_image[1] + moving_reg_image[0]
    lossmap_result = loss_lower_map + loss_upper_map

    loss_org, loss_warp, max_value = normalization_both(loss_map_org, lossmap_result)

    if save:
        plt.figure(figsize=(10.6, 2.5), dpi=300)
    else:
        plt.figure(figsize=(10.6, 2.5))

    color_bar = 'bone'

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'style': 'italic',
            'size': 10,
            }

    font2 = {'family': 'Times New Roman',
             'weight': 'bold',
             'style': 'italic',
             'size': 10,
             }

    a = [i for i in range(tmp_mask.shape[1])]
    b = [i for i in range(tmp_mask.shape[0])]
    X, Y = np.meshgrid(a, b)


    def resizeimage(image, resize=(194, 194), crop=False):
        h, w = image.shape
        croprange = 5
        if crop:
            image = image[0: h-croprange, croprange: w]
        image = cv2.resize(image, resize)
        return image

    # org
    plt.subplot(1, 6, 1)
    plt.imshow(resizeimage(moving_crop_image, crop=True), vmin=0, vmax=1, cmap='gray')
    plt.title('Moving', fontname="Times New Roman")
    plt.axis('off')

    upper = 'Upper: dz = {:.5f}, dθ = {:.5f}, dx = {:.5f}, dy = {:.5f}'.format((1 - upper_result[0] + 1),
                                                                               -upper_result[1],
                                                                               upper_result[2] * (length / 224),
                                                                               upper_result[3] * (length / 224))

    lower = 'Lower: dz = {:.5f}, dθ = {:.5f}, dx = {:.5f}, dy = {:.5f}'.format((1 - lower_result[0] + 1),
                                                                               -lower_result[1],
                                                                               lower_result[2] * (length / 224),
                                                                               lower_result[3] * (length / 224))

    plt.text(2, 215, upper, ha='left', wrap=True, fontdict=font)
    plt.text(2, 240, lower, ha='left', wrap=True, fontdict=font)

    plt.subplot(1, 6, 2)
    plt.imshow(resizeimage(moving_crop_image, crop=True), vmin=0, vmax=1, cmap='gray')
    plt.imshow(resizeimage(movin_mask, crop=True), cmap='GnBu', alpha=0.2)
    plt.title('Segmentation', fontname="Times New Roman")
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.imshow(resizeimage(fixed_org_image, crop=True), vmin=0, vmax=1, cmap='gray')
    plt.title('Fixed', fontname="Times New Roman")

    plt.axis('off')

    # result

    plt.subplot(1, 6, 4)
    plt.imshow(resizeimage(moving_result, crop=True), vmin=0, vmax=1, cmap='gray')
    plt.title('Warpped', fontname="Times New Roman")
    plt.axis('off')

    # loss
    norm = mpl.colors.Normalize(vmin=0, vmax=0.8)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.subplot(1, 6, 5)
    plt.imshow(resizeimage(loss_org, crop=True), norm=norm)
    plt.title('Original L2 Distance', fontname="Times New Roman")
    text = 'MSE$_{original}$' + ' = {:.5f}'.format(loss_val.detach().numpy())
    plt.text(2, 215, text, ha='left', wrap=True, fontdict=font)
    plt.axis('off')
    plt.subplot(1, 6, 6)
    plt.imshow(resizeimage(loss_warp, crop=True), norm=norm)
    plt.title('Warpped L2 Distance', fontname="Times New Roman")
    Loss = 'MSE$_{warpped}$' + ' = {:.5f}'.format(loss_result.detach().numpy())
    JSN = 'JSN = {:.5f}mm'.format(((upper_result - lower_result)[3] * (length / 224 * 0.175)))

    plt.text(0, 215, Loss, ha='left', wrap=True, fontdict=font2)
    plt.text(0, 240, JSN, ha='left', wrap=True, fontdict=font2)
    plt.axis('off')

    plt.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.95, wspace=0.03, hspace=0.2)
    file_name = output_path + '/{}.png'.format(filename)
    if save:
        # plt.savefig(file_name, transparent=False, dpi=300)

        # contain = [filename[:3], filename[4:7], filename[9], filename[13],
        #            upper_result[3] * (length / 224),
        #            lower_result[3] * (length / 224),
        #            ((upper_result - lower_result)[3] * (length / 224 * 0.175))]
        #
        # contain = ','.join(str(v) for v in contain)

        # f_test = open(output_path + 'JSN.txt', 'a')
        # f_test.writelines(contain)
        # f_test.write('\n')
        # f_test.close()

        # if loss_result < 0.002:
        #     file_name_3 = '../Result/Registration_image/normal/{}/{}.png'.format(joint, filename)
        #     plt.savefig(file_name_3, transparent=False, dpi=300)

        # if ((np.abs(upper_result[1]) > 0.05) or (np.abs(lower_result[1]) > 0.05)) and loss_result < 0.01:
        #     file_name_2 = '../Result/Registration_image/rotation' + '/{}.png'.format(filename)
        #     plt.savefig(file_name_2, transparent=False, dpi=300)
        # if ((np.abs(upper_result[0] - 1) > 0.05) or (np.abs(lower_result[0] - 1) > 0.05)) and loss_result < 0.01:
        #     file_name_2 = '../Result/Registration_image/scaling' + '/{}.png'.format(filename)
        #     plt.savefig(file_name_2, transparent=False, dpi=300)
        #     print('{} {} Scaling Get'.format(joint, filename))
        #     plt.clf()
        if loss_val > 0.02 and loss_result < 0.02:
            file_name_2 = '../Result/Registration_image/invasion' + '/{}.png'.format(filename)
            plt.savefig(file_name_2, transparent=False, dpi=300)
            print('{} {} Invasion Get'.format(joint, filename))
            plt.clf()
            return True
        else:
            return False

    else:
        plt.show()

if __name__ == "__main__":
    # root_path = '../Data/finger_joint_all_dataset/'
    root_path = '../Data/hongkong_data_2/test_L/'

    '''
    single picture
    '''
    output_path = '../Result/Registration_image/scaling'
    filename = '053_0_1_001'
    # 053_0_1_001 001_2_2_003
    filename2 = '053_0_1_002'
    joint = filename[4:7]
    # moving_path = root_path + 'moving/{}.bmp'.format(filename)
    # fixed_path = root_path + 'fixed/{}.bmp'.format(filename)
    moving_path = '../Data/hongkong_data_2/data_L/0/1_1/1_4.bmp'
    fixed_path = '../Data/hongkong_data_2/data_L/0/1_1/3_4.bmp'
    # moving_seg_path = root_path + 'moving_seg/{}.bmp'.format(filename2)
    # fixed_seg_path = root_path + 'fixed_seg/{}.bmp'.format(filename2)
    moving = np.array(Image.open(moving_path))
    fixed = np.array(Image.open(fixed_path))
    fixed = cv2.cvtColor(fixed, cv2.COLOR_RGB2GRAY)
    moving = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY)

    # moving_seg = np.array(Image.open(moving_seg_path))
    # fixed_seg = np.array(Image.open(fixed_seg_path))

    _, moving_seg = inputimage(moving)
    _, fixed_seg = inputimage(fixed)
    length = moving.shape[0]

    flag = predict(moving, fixed, moving_seg, fixed_seg, length, filename, save=False, output_path=output_path,
                   joint=joint)

    '''
    set picture
    '''
    image_list = os.listdir(root_path + 'moving')
    # image_list = hongkong_list
    output_path = '../Result/Registration_image/'
    # output_list = os.listdir(output_path)
    # with open(output_path + 'JSN.txt', 'a+', encoding='utf-8') as test:
    #     test.truncate(0)

    jointArr = ['0_0', '0_1', '1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '4_1', '4_2']
    jointArr = ['2_2', '3_1', '3_2', '4_1', '4_2']
    # select_list = ['123_0_0_002', '011_0_1_032', '113_0_1_019', '113_0_1_032', '113_0_1_025', '077_1_1_019',
    #                '159_1_1_000',
    #                '044_1_2_002', '097_1_2_001', '108_1_2_002', '113_1_2_033',
    #                '055_2_1_002', '011_2_2_011', '113_2_2_012', '113_2_2_004', '085_2_2_002', '068_2_2_002',
    #                '011_3_1_002', '046_3_1_001', '113_3_1_010', '113_3_1_012', '002_3_2_001', '010_3_2_020',
    #                '077_3_2_004', '181_3_2_000',
    #                '062_4_1_002', '077_4_1_014', '109_4_1_002', '184_4_1_015',
    #                '008_4_2_001', '083_4_2_001']
    # 063_1_2_000

    # for joint in jointArr:
    #     print(joint)
    #     count = 0
    #     for i in image_list:
    #         filename = i[:-4]
    #         joint_num = i[4:7]
    #
    #         if joint_num == joint:
    #             # if filename + '.png' not in output_list:
    #             # if filename in select_list:
    #                 moving_path = root_path + 'moving/{}.bmp'.format(filename)
    #                 fixed_path = root_path + 'fixed/{}.bmp'.format(filename)
    #                 moving_seg_path = root_path + 'moving_seg/{}.bmp'.format(filename)
    #                 fixed_seg_path = root_path + 'fixed_seg/{}.bmp'.format(filename)
    #                 moving = np.array(Image.open(moving_path))
    #                 fixed = np.array(Image.open(fixed_path))
    #                 moving_seg = np.array(Image.open(moving_seg_path))
    #                 fixed_seg = np.array(Image.open(fixed_seg_path))
    #                 # moving_org, moving_seg = inputimage_path(moving_path)
    #                 # fixed_org, fixed_seg = inputimage_path(fixed_path)
    #                 length = moving.shape[0]
    #                 flag = predict(moving, fixed, moving_seg, fixed_seg, length, filename, save=True, output_path=output_path, joint=joint)
    #                 if flag:
    #                     count += 1
    #                     print('Count:', joint, count)
    #         # if count == 20:
    #         #     break
