import cv2
import numpy as np
import scipy.fftpack
from scipy.optimize import leastsq
from skimage import morphology

def PIPOC(img1, img2, segmentation1, segmentation2, areaNum):
# def PIPOC(img1, img2, segmentation1, segmentation2, areaNum, src, ID, date, J):
    returnArr = []

    imgShape = img1.shape
    hy = np.hanning(imgShape[0])
    hx = np.hanning(imgShape[1])
    hw = hy.reshape(hy.shape[0], 1) * hx

    phase1 = scipy.fftpack.fft2(img1 * hw)
    phase1 = np.real(scipy.fftpack.ifft2(phase1 / np.abs(phase1)))
    phase2 = scipy.fftpack.fft2(img2 * hw)
    phase2 = np.real(scipy.fftpack.ifft2(phase2 / np.abs(phase2)))

    for i in range(areaNum):
        returnArr.append(poc(np.where(segmentation1 == i, phase1, 0), np.where(segmentation2 == i, phase2, 0)))
        # returnArr.append(poc(np.where(segmentation1 == i, phase1, 0), np.where(segmentation2 == i, phase2, 0), src, ID, date, J, i))
    return returnArr

def poc(img1, img2, fileName = False, mf = (7, 7)):
# def poc(img1, img2, src, ID, date, J, i,fileName = False, mf = (7, 7), ):
    center = tuple(np.array(img2.shape) / 2)
    m = np.floor(list(center))
    u = list(map(lambda x: x / 2.0, m))

    r = pocFun(img1, img2)
    r2 = (r / r.max() * 256)
    # cv2.imwrite(src + ID + '/300000/' + date + '_' + J + '_' + str(i) + '.bmp', r2)

    if fileName != False:
        x = (r/r.max()*300)
        cv2.imwrite(fileName, x)

    # least-square fitting
    max_pos = np.argmax(r)
    peak = (np.int(max_pos / img1.shape[1]), max_pos % img1.shape[1])

    if np.abs(peak[0]-m[0]) > 20 or np.abs(peak[1]-m[1]) > 20:
        return [[0, 0], 0]

    fitting_area = r[peak[0] - mf[0]: peak[0] + mf[0] + 1,
                     peak[1] - mf[1]: peak[1] + mf[1] + 1]

    p0 = [0.5, -(peak[0] - m[0]) - 0.02, -(peak[1] - m[1]) - 0.02]
    y, x = np.mgrid[-mf[0]:mf[0] + 1, -mf[1]:mf[1] + 1]
    y = y + peak[0] - m[0]
    x = x + peak[1] - m[1]
    errorfunction = lambda p: np.ravel(pocfunc_model(p[0], p[1], p[2], r, u)(y, x) - fitting_area)
    plsq = leastsq(errorfunction, p0)
    return ([[plsq[0][2], plsq[0][1]], plsq[0][0]])

def pocFun(img1, img2):

    # compute 2d fft
    F = scipy.fftpack.fft2(img1)
    F = F /np.abs(F)
    G = scipy.fftpack.fft2(img2)
    G = G /np.abs(G)
    G_ = np.conj(G)
    R = F * G_

    return scipy.fftpack.fftshift(np.real(scipy.fftpack.ifft2(R)))


def pocfunc_model(alpha, delta1, delta2, r, u):
    N1, N2 = r.shape
    V1, V2 = list(map(lambda x: 2 * x + 1, u))
    return lambda n1, n2: alpha / (N1 * N2) * np.sin((n1 + delta1) * V1 / N1 * np.pi) * np.sin((n2 + delta2) * V2 / N2 * np.pi)\
                                            / (np.sin((n1 + delta1) * np.pi / N1) * np.sin((n2 + delta2) * np.pi / N2))





'''
# 沟壑检测图像        gully detection function (Fig.2) (Fig.1 step 4)
# 生成沟壑概率图       task: detect the depth of each point
# img 输入图像（灰度图）     parameter: img: input image(gray)
# 返回值 沟壑概率图     return: gully depth map (Detailed description you can find in paper)
'''
def gullyDetect(img, maxWidth = 7):
    # 图像大小      image size
    imgShape = img.shape
    imgShape = (imgShape[1], imgShape[0])
    # 横向相加      filter (Paper Fig.2 (b))
    horizon = img + cv2.warpAffine(img, np.float32([[1, 0, 1], [0, 1, 0]]), imgShape) / 2 + cv2.warpAffine(img, np.float32([[1, 0, -1], [0, 1, 0]]), imgShape) / 2
    # 高度为3的检测窗口     Upper part and lower part of gully depth map with 3pixel width
    medArr1 = cv2.warpAffine(horizon, np.float32([[1, 0, 0], [0, 1, 1]]), imgShape) - horizon
    medArr2 = cv2.warpAffine(horizon, np.float32([[1, 0, 0], [0, 1, -1]]), imgShape) - horizon
    # 沟壑概率图像        gully depth map with 3pixel width (Paper Fig.2 (c) g3(x,y))
    resArr = np.where(medArr1 > medArr2, medArr2, medArr1)
    # cv2.imwrite('training data/jointData/4_2/2016.08.01_L_0.bmp', resArr*6)
    # 缩放高度 进行检测     gully depth map with any width
    vertical1 = horizon.copy()
    vertical2 = horizon.copy()
    medArr = horizon.copy()
    for i in range(maxWidth//2):
        # 纵向合并加快运算      Upper part and lower part of gully depth map
        vertical1 = cv2.warpAffine(vertical1, np.float32([[1, 0, 0], [0, 1, 1]]), imgShape)
        vertical2 = cv2.warpAffine(vertical2, np.float32([[1, 0, 0], [0, 1, -1]]), imgShape)
        medArr = medArr + vertical1 + vertical2
        # 当前检测窗口高度      gully depth map width i
        num = (i * 2 + 3)
        medArr1 = cv2.warpAffine(medArr, np.float32([[1, 0, 0], [0, 1, num]]), imgShape) - medArr
        medArr2 = cv2.warpAffine(medArr, np.float32([[1, 0, 0], [0, 1, -num]]), imgShape) - medArr
        medArr3 = np.where(medArr1 > medArr2, medArr2, medArr1) / num
        # cv2.imwrite('d01-04-Parts/17112118_d01-04-distance_'+str(i*2+3)+'.jpg', medArr3)
        resArr = np.where(resArr > medArr3, resArr, medArr3)
    # 精度为小数点后一位 过长的精度可能导致计算错误       The accuracy of the result is one decimal place
    # cv2.imwrite('training data/jointData/2016.08.01_L_A.bmp', resArr * 6)
    # cv2.imwrite('2.jpg', resArr * 3)
    resArr = resArr * 10
    resArr = resArr.astype(np.int64)
    return resArr


'''
# 根据沟壑概率图对图像进行横向分割      image segmentation (Fig.3)(Fig.1 step 4)
# gullyPro 输入的沟壑概率图     parameter: gullyPro: gully depth map
# 返回值 分割为两部分的图像     return: upper and lower bones
'''
def separation(gullyPro):
    # 图像转置方便计算      image transposition for calculation
    integralArr = gullyPro.copy().T
    resArrT = integralArr.copy()
    # 返回矩阵      return array
    areaArr = np.zeros(gullyPro.shape, np.uint8).T              # 分割区域矩阵
    routeArr = []                                               # 分割线数组
    # routebcArr = []                                             # 用于生成背景值
    # 滤波矩阵      filter matrix
    smsquare = morphology.square(3)
    # 膨胀叠加      morphology.dilation
    for i in range(1, len(integralArr)):
        integralArr[i] = morphology.dilation(np.array([integralArr[i - 1]]), smsquare)[0] + integralArr[i]
    # 根据最后一列最大值 推算路径        Estimate the path based on the maximum value of each column in integralArr
    maxNum = np.max(integralArr[i])
    routeIndex = np.where(integralArr[i] == maxNum)[0][0]
    nextNum = maxNum - resArrT[i][routeIndex]
    # routeIndex = 300
    # nextNum = integralArr[i][300] - resArrT[i][routeIndex]
    areaArr[i][routeIndex] = 255
    routeArr.append(routeIndex)
    # routebcArr.append(img[i][routeIndex])
    # 路径推算      Path estimation
    for i in range(i, 0, -1):
        routeIndexArr = np.where(integralArr[i - 1] == nextNum)[0]
        # 多条路径存在        multiple paths
        if len(routeIndexArr) > 1:
            minDistance = 2
            for index in routeIndexArr:
                if np.abs(index - routeIndex) < minDistance:
                    routeIndex = np.array([index])
                    minDistance = np.abs(index - routeIndex)
        else:
            routeIndex = routeIndexArr
        nextNum = nextNum - resArrT[i - 1][routeIndex]
        # 分割线信息     segmentation curve
        routeArr.append(routeIndex[0])
        # 分割线标识
        areaArr[i][routeIndex] = 255
        # 上方区域标识
        areaArr[i, :routeIndex[0]] = 1
    # 分割结果
    return [areaArr.T, routeArr]
