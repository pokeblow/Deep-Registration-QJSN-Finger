import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

class Data_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.img_path = glob.glob(os.path.join(data_path, 'moving/*.bmp'))
        self.transform = transform

    def __getitem__(self, index):
        # 数据路径
        moving_path = self.img_path[index]
        fixed_path = moving_path.replace('moving', 'fixed')
        moving_mask_path = moving_path.replace('moving', 'moving_seg')
        fixed_mask_path = moving_path.replace('moving', 'fixed_seg')

        # 读取数据
        moving = np.array(Image.open(moving_path))
        fixed = np.array(Image.open(fixed_path))
        moving_seg = np.array(Image.open(moving_mask_path))
        fixed_seg = np.array(Image.open(fixed_mask_path))

        # 随机旋转和缩放
        random_num = random.randint(0, 10)
        if random_num > 6:
            h, w = moving.shape

            angle = random.randint(-8, 8) / 2

            center = (w / 2, h / 2)
            scale = float("%.2f" % random.uniform(0, 2))

            M_R = cv2.getRotationMatrix2D(center, angle, scale)

            print(M_R)

            moving = cv2.warpAffine(moving, M_R, (h, w))
            moving_seg = cv2.warpAffine(moving_seg, M_R, (h, w))

            plt.imshow(fixed_seg)
            plt.show()

        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range
        moving = normalization(moving)
        fixed = normalization(fixed)

        # transform
        moving = self.transform(Image.fromarray(moving))
        fixed = self.transform(Image.fromarray(fixed))
        moving_seg = self.transform(Image.fromarray(moving_seg))
        fixed_seg = self.transform(Image.fromarray(fixed_seg))
        # label = torch.tensor(label)

        return moving, fixed, moving_seg, fixed_seg

    def __len__(self):
        return len(self.img_path)

if __name__ == "__main__":
    data_path = 'Data/test_dataset'
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor
                                    ])
    image_dataset = Data_Loader(data_path, transform)
    print("Length of Dataset:", len(image_dataset))



