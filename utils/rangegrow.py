import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool


class Rangegrow():
    def getGrayDiff(self, img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))

    def selectConnects(self, p):
        if p != 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                        Point(0, 1), Point(-1, 1), Point(-1, 0)]
        else:
            connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return connects

    def regionGrow(self, img, seeds, thresh, p):
        height, weight = img.shape
        seedMark = np.zeros(img.shape)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        label = 1
        connects = self.selectConnects(p)
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)

            seedMark[currentPoint.x, currentPoint.y] = label
            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                    continue
                grayDiff = self.getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                    seedMark[tmpX, tmpY] = label
                    seedList.append(Point(tmpX, tmpY))
        return seedMark

    def getresult(self, image, size, set):
        seeds = [Point(500, 500), Point(10, 10), Point(500, 10), Point(10, 500)]
        binaryImg = self.regionGrow(image, seeds, size, 1)
        binaryImg = (binaryImg == 0)
        image[(binaryImg == 0)] = set
        return image

    def getresultchannel3(self, image):
        image = image.transpose((2, 0, 1))
        result = np.array([self.getresult(image[0], 3, 0), self.getresult(image[1], 5, 0), self.getresult(image[2], 2, 0)])
        result = result.transpose((1, 2, 0))
        return result


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


if __name__ == '__main__':
    input = cv2.imread('/Users/wanghaolin/PycharmProjects/Unet/data/train/image/000002.jpg', 0)
    input2 = cv2.imread('/Users/wanghaolin/PycharmProjects/Unet/data/train/image/000002.jpg')
    rangegrow = Rangegrow()
    result = rangegrow.getresultchannel3(input2)
    print(result.shape)
