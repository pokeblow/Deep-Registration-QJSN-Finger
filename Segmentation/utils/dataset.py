import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from utils.rangegrow import Rangegrow
from torchvision import transforms


class ISBI_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.bmp'))
        self.transform = transform

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))
        # label = label.transpose((1, 0))
        label[label != 0] = 1

        plt.imshow(label)
        plt.show()

        # h, w = label_crop.shape
        # image = image.transpose((2, 0, 1))
        # image_channel1 = np.array(image[0][0:h, 0:w])
        # image_channel2 = np.array(image[1][0:h, 0:w])
        # image_channel3 = np.array(image[2][0:h, 0:w])
        # image = np.array([image_channel1, image_channel2, image_channel3])
        # image = image.transpose((1, 2, 0))

        # print(image.shape)
        # print(label_crop.shape)

        # plt.imshow(image)
        # plt.imshow(label_crop, alpha=0.5)
        # plt.show()

        if self.transform:
            image = self.transform(Image.fromarray(image))
            label = self.transform(Image.fromarray(label))

        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range * 255

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("Data/hip_space_segment/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=False)
    for image, label in train_loader:
        print(image.shape)
        print(label.shape)
