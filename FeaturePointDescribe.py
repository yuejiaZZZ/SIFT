import numpy as np
import cv2 as cv
from FeaturePointExtract import *

"""************************ step3：关键点的主方向计算 **************************"""

"""************************ step4：关键点的描述 **************************"""


if __name__ == '__main__':
    img = cv.imread('data/Lena.tif', 0)

    S = 3        #S表示提取特征点的层数
    sigma_0 = 1.6         #sigma为模糊尺度
    DOG_pyramid = DoGPyramid(img, sigma_0, S)
    key_points = np.load('results/key_points.npy')
    m = 2