import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import cv2
import utils.retinex as retinex

import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import utils.library as lb

from segmentation import Segmentation

from model.model_ResCNN_mask_3 import Registration_ResCNN_3


class Registration():
    def __init__(self, moving, fixed, moving_seg=None, fixed_seg=None, seg_model='Gully'):
        self.moving = moving
        self.fixed = fixed
        self.org_size = moving.shape[0]

        if (moving_seg is None) and (fixed_seg is None):
            seg_m = Segmentation(moving)
            seg_f = Segmentation(fixed)
            if seg_model == 'Gully':
                self.moving_seg = seg_m.seg_Gully(kernel_size=1)
                self.fixed_seg = seg_f.seg_Gully(kernel_size=1)
            else:
                self.moving_seg = seg_m.seg_Unet()
                self.fixed_seg = seg_f.seg_Unet()
        else:
            self.moving_seg = moving_seg
            self.fixed_seg = fixed_seg

        self.moving, self.moving_seg = self.inputimage(self.moving, self.moving_seg)
        self.fixed, self.fixed_seg = self.inputimage(self.fixed, self.fixed_seg)


        self.loss_org_map = None
        self.loss_warp_map = None
        self.loss_org = 0
        self.loss_warp = 0
        self.moving_org = None
        self.fixed_org = None
        self.moving_warp = None
        self.tmp_mask = None
        self.result = []
        self.JSN = 0


    def normalization_both(self, image_1, image_2):
        range = np.max(image_1) - np.min(image_1)
        image_1 = (image_1 - np.min(image_1)) / range
        image_2 = (image_2 - np.min(image_2)) / range
        return image_1, image_2, np.max(image_1)


    def inputimage(self, image, segmentation):
        segmentation = lb.separation(lb.gullyDetect(image))[0]

        segmentation[segmentation != 0] = 1

        range_size = 4

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
        segmentation = segmentation[range_size: h - range_size, range_size: w - range_size]
        image = image[range_size: h - range_size, range_size: w - range_size]

        return image, segmentation


    def register(self, loss_out=False, parameter_out=False):
        moving = self.moving
        fixed = self.fixed
        moving_seg = self.moving_seg
        fixed_seg = self.fixed_seg
        # 选择设备，有cuda用cuda，没有就用cpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。
        net = Registration_ResCNN_3()
        # 加载模型参数
        # best_model_0127 best_model_0217_P
        net.load_state_dict(torch.load('../JSNmeasurement/parameter/best_model_0301.pth', map_location='cpu'))
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

        loss_org, loss_warp, max_value = self.normalization_both(loss_map_org, lossmap_result)


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
                image = image[0: h - croprange, croprange: w]
            image = cv2.resize(image, resize)
            return image

        # org
        plt.subplot(1, 6, 1)
        plt.imshow(resizeimage(moving_crop_image, crop=True), vmin=0, vmax=1, cmap='gray')
        plt.title('Moving', fontname="Times New Roman")
        plt.axis('off')

        upper = 'Upper: dz = {:.5f}, dθ = {:.5f}, dx = {:.5f}, dy = {:.5f}'.format((1 - upper_result[0] + 1),
                                                                                   -upper_result[1],
                                                                                   upper_result[2] * (self.org_size / 224),
                                                                                   upper_result[3] * (self.org_size / 224))

        lower = 'Lower: dz = {:.5f}, dθ = {:.5f}, dx = {:.5f}, dy = {:.5f}'.format((1 - lower_result[0] + 1),
                                                                                   -lower_result[1],
                                                                                   lower_result[2] * (self.org_size / 224),
                                                                                   lower_result[3] * (self.org_size / 224))

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
        JSN = 'JSN = {:.5f}mm'.format(((upper_result - lower_result)[3] * (self.org_size / 224 * 0.175)))

        plt.text(0, 215, Loss, ha='left', wrap=True, fontdict=font2)
        plt.text(0, 240, JSN, ha='left', wrap=True, fontdict=font2)
        plt.axis('off')

        plt.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=0.95, wspace=0.03, hspace=0.2)

        return ((upper_result - lower_result)[3] * (self.org_size / 224 * 0.175)), (upper_result[3] * (self.org_size / 224 * 0.175)), (lower_result[3] * (self.org_size / 224 * 0.175))

    def show_image(self):
        plt.show()

    def save_image(self, path):
        file_name = path
        plt.savefig(file_name, transparent=False, dpi=300)
        # print('Save Done')
        plt.clf()