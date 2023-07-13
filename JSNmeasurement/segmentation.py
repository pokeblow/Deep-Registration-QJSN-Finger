import numpy as np
import torch
from model.unet_model import ResNet34UnetPlus
from utils.rangegrow import *
import utils.library as lb
from torchvision import transforms
from PIL import Image
import utils.retinex as retinex


class Segmentation():
    def __init__(self, image):
        self.image = image

    def inputimage(self):
        if self.image.shape[0] > 200:
            image = cv2.resize(self.image, (100, 100))
        else:
            image = self.image

        h, w = image.shape
        range = 4

        image = image[range: h - range, range: w - range]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = retinex.MSRCP(
            img=image,
            sigma_list=[15, 80, 200],
            low_clip=0.01,
            high_clip=0.99,
        )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.medianBlur(image, 1)

        return image

    def seg_Unet(self, confidence_range=0.98, kernel_size=5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = ResNet34UnetPlus(num_channels=1, num_class=1)
        net.to(device=device)
        net.load_state_dict(torch.load('./JSNmeasurement/parameter/best_model_seg.pth', map_location=device), strict=False)
        net.eval()

        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor()])

        image = self.inputimage()
        image = transform(Image.fromarray(image))
        image = np.array(image)
        image = np.array([image])
        img_tensor = torch.from_numpy(image)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])
        pred = pred[0]

        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        pred = normalization(pred)
        pred[pred < confidence_range] = 0
        pred[pred >= confidence_range] = 1

        seeds = [Point(5, 250), Point(5, 5)]
        pred = regionGrow(pred, seeds, 1)

        pred = -pred
        pred[pred == -1] = 1

        seeds = [Point(250, 250), Point(250, 5)]
        pred = regionGrow(pred, seeds, 1)

        pred = pred - 1
        pred[pred == -1] = 1

        pred = self.soomth(pred, kernel_size=kernel_size)

        pred = cv2.resize(pred, (self.image.shape))

        return pred

    def soomth(self, mask, kernel_size):
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

    def seg_Gully(self, kernel_size=3):
        if self.image.shape[0] > 200:
            image = cv2.resize(self.image, (100, 100))
        else:
            image = self.image
        segmentation = lb.separation(lb.gullyDetect(image))[0]
        segmentation[segmentation != 0] = 1
        h, w = segmentation.shape
        for i in range(h):
            if segmentation[i][1] == 0:
                break
        for j in range(i):
            segmentation[j][0] = 1

        # segmentation = self.soomth(segmentation, kernel_size=kernel_size)

        return segmentation
