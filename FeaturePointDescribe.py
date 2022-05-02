import numpy as np
import cv2 as cv
from FeaturePointExtract import *

"""************************ step3：关键点的主方向计算 **************************"""
# 3.1 根据关键点确定尺度sigma
def ComputeSigma(key_point_coor):
    """
    返回当前特征点对应的模糊尺度
    :param sigma_0: 初始模糊尺度
    :param S: 表示提取特征点的层数
    :param key_point_coor: 一个特征点数组，ndarray对象，key_point = [value, octave, i, j, s]
    :return: 特征点的模糊尺度
    """
    octave = key_point_coor[0]
    s = key_point_coor[3]
    sigma = sigma_0 * 2 ** (octave + s/S)
    return sigma

# 3.2 计算特征点梯度的幅值和方向
def ComputeGrad(L, i, j):
    """
    输出该特征点梯度的幅值和方向
    :param DOG_pyramid: DOg金字塔
    :param key_point_coor: 关键点坐标，只有坐标信息[o,i,j,s]
    :return: 该特征点的幅值和方向
    """
    opposite_edge = L[i+1,j] - L[i-1, j]   # y
    near_edge = L[i, j+1] - L[i, j-1]   # x
    magnitude = np.sqrt(opposite_edge **2 + near_edge ** 2)
    theta = np.arctan2(opposite_edge, near_edge) + np.pi   # theta \in [0,2pi]
    return magnitude, theta

# 3.3 以特征点为中心，3*1.5sigma为半径的区域内，统计特征点附近的梯度方向直方图
def ComputeDirectionHist(DOG_pyramid, key_point_coor, sigma, num_bins=36):
    """
    返回该特征点的邻域的梯度方向直方图
    :param DOG_pyramid:
    :param key_point_coor: 一个关键点坐标，只有坐标信息[o,i,j,s]
    :param sigma: 特征点的模糊尺度
    :param num_bins: 直方图柱数
    :return: 梯度方向直方图
    """
    octave = key_point_coor[0]
    i = key_point_coor[1]
    j = key_point_coor[2]
    s = key_point_coor[3]
    whole_octave = DOG_pyramid[octave]
    L = whole_octave[:, :, s]
    R = int(np.round(3*1.5*sigma))
    hist = np.zeros(num_bins)
    denominator = 2 * ((1.5 * sigma) ** 2)
    for m in range(-1*R, R+1):
        for n in range(-1*R, R+1):
            m_index = i + m
            n_index = j + n
            if 0<m_index<L.shape[0]-1 and 0<n_index<L.shape[1]-1:
                magnitude, theta = ComputeGrad(L, m_index, n_index)
                theta_delta = 2*np.pi/num_bins
                bin = int(np.floor(theta/theta_delta))
                if 0 <= bin < num_bins:
                    magnitude_weight = np.exp(-(m*m + n*n)/denominator) # 幅度加权，高斯核用1.5sigma
                    hist[bin] = hist[bin] + magnitude_weight * magnitude
                else:
                    print("Bin:{}".format(bin))
                    print("角度超出统计区间，{}".format(theta))

    # 平滑直方图
    smooth_hist = np.zeros(num_bins)
    temp_hist = np.zeros(num_bins+2)
    temp_hist[1:num_bins+1] = hist
    for i in range(num_bins):
        smooth_hist[i] = 0.25 * temp_hist[i] + 0.5 * temp_hist[i+1] +0.25 * temp_hist[i+2]

    return smooth_hist



# 3.4 输出含有位置、尺度、方向信息的特征点
def ParabolicInterp(hist, center):
    """
    抛物线插值，输入为直方图和中心点坐标，输出为精确的角度
    :param hist: 直方图
    :param center: 中心坐标
    :return: 精确角度
    """
    num_hist = len(hist)
    if center==0 or center==35:
        exact_direction = 0
    else:
        x = [center-1, center, center+1]
        y = [hist[center-1], hist[center], hist[center+1]]
        n = len(x)
        a = np.zeros([n,n])
        for i in range(n) :
            for j in range(n) :
                a[i,j]=x[i]**(n-1-j)
        para = np.linalg.solve(a, y)
        results =-para[1]/2/para[0]
        exact_direction = results * 2 * np.pi/num_hist
    return exact_direction

def OutKeyPointsWithDirection(DOG_pyramid,key_points, key_points_coor):
    """
    输出有位置、尺度、方向信息的特征点列表
    :param DOG_pyramid:
    :param key_points: 特征点列表
    :param key_points_coor: 特征点坐标列表
    :return: 有位置、尺度、方向信息的特征点 key_points_direction[k] = [Value, octave, i, j, s, direction],
            direction是弧度制
    """
    key_points_direction = []
    for k in range(len(key_points_coor)):
        key_point = key_points[k]
        key_point_coor = key_points_coor[k, :]
        sigma = ComputeSigma(key_point_coor)
        direction_hist = ComputeDirectionHist(DOG_pyramid, key_point_coor, sigma)
        num_bins = len(direction_hist)
        magnitude_max = np.max(direction_hist)
        magnitude_index = np.asarray(np.where(magnitude_max == direction_hist))
        threshold = magnitude_max * 0.8
        for n in range(num_bins):
            # 寻找主方向和辅方向，并通过抛物线插值得到精确值
            if (direction_hist[n] > threshold and (n < magnitude_index-1 or n > magnitude_index + 1)) or (n == magnitude_index):
                direction = ParabolicInterp(direction_hist, n)
                temp_key_point = np.hstack((key_point, direction))
                key_points_direction.append(temp_key_point)
    key_points_direction = np.asarray(key_points_direction)
    return key_points_direction

"""************************ step4：关键点的描述 **************************"""
def ComputeDescriptorHist(correct_feature_map, start_m_index, start_n_index, Bp, num_bins=8):
    descriptor_hist = np.zeros(num_bins)
    for i in range(Bp):
        for j in range(Bp):
            m_index = start_m_index + i
            n_index = start_n_index + j
            if m_index > 0 and n_index > 0and m_index < correct_feature_map.shape[0]-1 and n_index < correct_feature_map.shape[1] -1:

                magnitude, theta = ComputeGrad(correct_feature_map, m_index, n_index)
                theta_delta = 2 * np.pi / num_bins

                bin = int(np.floor(theta/theta_delta))
                if bin == 8:
                    bin = 0
                if 0 <= bin < num_bins:
                    # 幅值分解加权
                    weight_l2 = (theta + np.pi - bin * np.pi / 4) / (np.pi / 4)
                    weight_l1 = 1 - weight_l2
                    descriptor_hist[bin] = descriptor_hist[bin] + weight_l1 * magnitude
                    if bin == num_bins - 1:
                        descriptor_hist[0] = descriptor_hist[0] + weight_l2 * magnitude
                    else:
                        descriptor_hist[bin+1] = descriptor_hist[bin+1] + weight_l2 * magnitude
                else:
                    print("bin:{}".format(bin))
                    print("超出统计范围，角度为：{}".format(theta))
    return descriptor_hist

def CorrectDirection(DOG_pyramid, key_point_direction):
    """
    根据当前特征点的方向，对特征图旋转
    :param DOG_pyramid: 特征金字塔
    :param key_point_direction: key_point_direction = [value, octave, i, j, s, direction]
    :return: 旋转后的特征图
    """
    # key_point_direction = [value, octave, i, j, s, direction]
    octave = np.int(key_point_direction[1])
    i = np.int(key_point_direction[2])
    j = np.int(key_point_direction[3])
    s = np.int(key_point_direction[4])
    theta_degree = (180 * key_point_direction[5]) / np.pi
    whole_octave = DOG_pyramid[octave]
    L = whole_octave[:, :, s]
    rotate_center = (j, i)   # (列，行)
    M = cv.getRotationMatrix2D(rotate_center, -1 * theta_degree, scale=1)   # 负为顺时针
    correct_feature_map = cv.warpAffine(L, M, (L.shape[0], L.shape[1]))
    # cv.imshow('sift', L)
    # cv.waitKey()
    # cv.imshow('rotate', correct_feature_map)
    # cv.waitKey()
    return correct_feature_map

def ComputeFeatureVector(correct_feature_map, key_point_direction):
    octave = np.int(key_point_direction[1])
    i = np.int(key_point_direction[2])
    j = np.int(key_point_direction[3])
    s = np.int(key_point_direction[4])
    Bp = np.int(4)
    R = np.int(8)

    feature_vector_list = []
    for m in range(np.int(-1*R), np.int(R/2+1), Bp):
        for n in range(np.int(-1*R), np.int(R/2+1), Bp):
            m_index = m + i
            n_index = n + j
            if m_index > 0 and n_index > 0 and m_index < correct_feature_map.shape[0]-1 and n_index < correct_feature_map.shape[1] -1:
                descriptor_hist = ComputeDescriptorHist(correct_feature_map, m_index, n_index, Bp)
                descriptor_hist_list = list(descriptor_hist)
                feature_vector_list += descriptor_hist_list
            else:
                descriptor_hist = np.zeros(8)
                descriptor_hist_list = list(descriptor_hist)
                feature_vector_list += descriptor_hist_list
    feature_vector = np.asarray(feature_vector_list)
    feature_vector = np.reshape(feature_vector, -1)
    return feature_vector

def GenerateDescriptor(DOG_pyramid, key_points_direction):
    num_key_points = len(key_points_direction)
    descriptor_list = []
    descriptor_coor = []
    for k in range(num_key_points):
        key_point_direction = key_points_direction[k]
        # 矫正主方向
        correct_feature_map = CorrectDirection(DOG_pyramid, key_point_direction)
        feature_vector = ComputeFeatureVector(correct_feature_map, key_point_direction)
        descriptor_list.append(feature_vector)
        descriptor_coor.append(key_point_direction[0:4])
    sift_descriptor = np.asarray(descriptor_list)
    descriptor_coor = np.asarray(descriptor_coor)
    return sift_descriptor, descriptor_coor

S = 3        #S表示提取特征点的层数
sigma_0 = 1.6         #sigma为模糊尺度
# if __name__ == '__main__':
#     img = cv.imread('data/Lena.tif', 0)
#
#     S = 3        #S表示提取特征点的层数
#     sigma_0 = 1.6         #sigma为模糊尺度
#     DOG_pyramid = DoGPyramid(img, sigma_0, S)
#     key_points = np.load('results/key_points.npy')
#
#     # 关键点的主方向计算，因为有的点方向不唯一，所以点计算方向后，点数增加
#     key_points_coor =key_points[:, 1:].astype(np.int)
#     key_points_direction = OutKeyPointsWithDirection(DOG_pyramid,key_points, key_points_coor)
#
#     # 生成特征点的描述子
#     sift_descriptor, descriptor_coor = GenerateDescriptor(DOG_pyramid, key_points_direction)