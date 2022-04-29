import numpy as np
import cv2 as cv
from FeaturePointExtract import *
from FeaturePointDescribe import *

if __name__ == '__main__':
    img = cv.imread('data/Lena.tif', 0)

    S = 3        #S表示提取特征点的层数
    sigma_0 = 1.6         #sigma为模糊尺度

    """*************************特征点提取*************************"""
    DOG_pyramid = DoGPyramid(img, sigma_0, S)

    # 保存特征点
    # key_points = DetectKeyPoint(DOG_pyramid)
    # file = open('results/key_points_list.txt', 'w')
    # for fp in key_points:
    #     file.write(str(fp))
    #     file.write('\n')
    # file.close()
    # np.save('results/key_points', key_points)

    key_points = np.load('results/key_points.npy')

    # 保存提取特征点后的图像
    # image = DrawFeaturePoint(img, key_points)
    # cv.imwrite('results/sift_feature_point.jpg', image)

    """*************************特征点描述*************************"""


