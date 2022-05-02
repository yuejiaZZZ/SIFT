import numpy as np
import cv2 as cv
from FeaturePointExtract import *
from FeaturePointDescribe import *

if __name__ == '__main__':

    """********* 读取两个图像 ******"""
    img_one = cv.imread('data/test_one.jpg', 0)
    img_two = cv.imread('data/test_two.jpg', 0)

    S = 3        #S表示提取特征点的层数
    sigma_0 = 1.6         #sigma为模糊尺度

    # """*********************************生成描述子****************************************"""
    # """*************************特征点提取*************************"""
    # DOG_pyramid_one = DoGPyramid(img_one, sigma_0, S)
    # key_points_one = DetectKeyPoint(DOG_pyramid_one)
    # np.save('results/key_points_one', key_points_one)
    #
    # DOG_pyramid_two = DoGPyramid(img_two, sigma_0, S)
    # key_points_two = DetectKeyPoint(DOG_pyramid_two)
    # np.save('results/key_points_two', key_points_two)
    #
    # """*************************特征点描述*************************"""
    # # 关键点的主方向计算，因为有的点方向不唯一，所以点计算方向后，点数增加
    # key_points_coor_one = key_points_one[:, 1:].astype(np.int)
    # key_points_direction_one = OutKeyPointsWithDirection(DOG_pyramid_one, key_points_one, key_points_coor_one)
    #
    # key_points_coor_two = key_points_two[:, 1:].astype(np.int)
    # key_points_direction_two = OutKeyPointsWithDirection(DOG_pyramid_two, key_points_two, key_points_coor_two)
    #
    # # 生成特征点的描述子
    # sift_descriptor_one, descriptor_coor_one = GenerateDescriptor(DOG_pyramid_one, key_points_direction_one)
    # np.save('results/sift_descriptor_one', sift_descriptor_one)
    # np.save('results/descriptor_coor_one', descriptor_coor_one)
    #
    # sift_descriptor_two, descriptor_coor_two= GenerateDescriptor(DOG_pyramid_two, key_points_direction_two)
    # np.save('results/sift_descriptor_two', sift_descriptor_two)
    # np.save('results/descriptor_coor_two', descriptor_coor_two)

    """*******************************载入数据****************************************"""
    DOG_pyramid_one = DoGPyramid(img_one, sigma_0, S)
    DOG_pyramid_two = DoGPyramid(img_two, sigma_0, S)
    key_points_one = np.load('results/key_points_one.npy')
    key_points_two = np.load('results/key_points_two.npy')

    image_one = DrawFeaturePoint(img_one, key_points_one)
    cv.imwrite('results/key_point_one.jpg', image_one)
    image_two = DrawFeaturePoint(img_two, key_points_two)
    cv.imwrite('results/key_point_two.jpg', image_two)

    sift_descriptor_one = np.load('results/sift_descriptor_one.npy')
    descriptor_coor_one = np.load('results/descriptor_coor_one.npy')
    sift_descriptor_two = np.load('results/sift_descriptor_two.npy')
    descriptor_coor_two = np.load('results/descriptor_coor_two.npy')
    print("第一张图提取到{}个sift描述子".format(len(sift_descriptor_one)))
    print("第二张图提取到{}个sift描述子".format(len(sift_descriptor_one)))

    image = np.concatenate([img_one, img_two], axis=1)

    num_match_point = 0
    for i in range(len(sift_descriptor_one)):
        # [value, o, i, j, s]
        dist_threshold = 0.5
        indicate_nearest = -1
        for j in range(len(sift_descriptor_two)):
            feature_one = sift_descriptor_one[i]
            feature_two = sift_descriptor_two[j]
            dist = np.linalg.norm(feature_one / np.linalg.norm(feature_one) - feature_two / np.linalg.norm(feature_two))
            if dist < dist_threshold:
                dist_threshold = dist
                indicate_nearest = j

        if indicate_nearest >= 0:
            num_match_point +=1
            x1 = np.int(descriptor_coor_one[i, 3])
            y1 = np.int(descriptor_coor_one[i, 2])
            x2 = np.int(descriptor_coor_two[indicate_nearest, 3])
            y2 = np.int(descriptor_coor_two[indicate_nearest, 2])

            point_one = (x1, y1)
            point_two = (x2+img_one.shape[1], y2)
            cv.circle(image, point_one, radius=2, color=(255, 0, 0))
            cv.circle(image, point_two, radius=2, color=(0, 255, 0))
            cv.line(image, point_one, point_two, color=(0, 0, 255), thickness=1)

    print("匹配的特征点有：{}个".format(num_match_point))
    cv.imshow("feature_point_match", image)
    cv.waitKey()
    cv.imwrite("results/feature_point_match.jpg", image)









