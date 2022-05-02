import numpy as np
import cv2 as cv

"""******************** step1: 多尺度（DoG）空间极值点检测 ***********************"""

def padding(image, kernel_size):
    """
    对原始图像根据卷积核做padding
    :param image: 输入图像
    :param kernel_size: 卷积核大小
    :return: padding后的图像
    """
    L = image.shape[0]
    H = image.shape[1]
    padding_image = np.zeros([L+kernel_size-1, H+kernel_size-1])
    adding_pixel = np.int((kernel_size-1)/2)
    padding_image[adding_pixel:L+adding_pixel, adding_pixel:H+adding_pixel] = image
    return padding_image

def conv2d(image, kernel):
    """
    二维卷积实现
    :param image: 输入图像
    :param kernel: 卷积核
    :return: 卷积输出
    """
    kernel_size = kernel.shape[0]
    padding_image = padding(image, kernel.shape[0])
    img_after_conv = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = padding_image[i:i+kernel_size, j:j+kernel_size]
            img_after_conv[i, j] = np.sum(temp*kernel)
    return img_after_conv

def MaxMinNorm(array):
    """
    对输入矩阵进行最大最小归一化
    :param array: 矩阵
    :return: 归一化后矩阵
    """
    array_min = np.min(array)
    array_max= np.max(array)
    norm_array = np.zeros(array.shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            norm_array[i,j] = (array[i,j]-array_min)/(array_max-array_min)
    return norm_array

def DownSample(image):

    L,H =image.shape
    image_after_downsample = np.zeros([int(L/2), int(H/2)])
    for i in range(int(L/2)):
        for j in range(int(H/2)):
            image_after_downsample[i, j] = image[i*2, j*2]
    return image_after_downsample

def GaussianKernel(sigma, kernel_shape):
    """
    返回一个指定模糊系数sigma的高斯核
    :param sigma: 高斯核参数sigma
    :return: 高斯核
    """
    x0 = int(round(kernel_shape/2))
    y0 = int(round(kernel_shape/2))
    X = np.linspace(1, kernel_shape, kernel_shape)
    Y = np.linspace(1, kernel_shape, kernel_shape)
    x, y = np.meshgrid(X, Y)
    gaussian_kernel = (1 / 2*np.pi*(sigma**2)) * np.exp(- ((x-x0)**2 +(y-y0)**2)/(2 * (sigma**2)))
    gaussian_kernel = gaussian_kernel/np.sum(gaussian_kernel)
    return gaussian_kernel

def GenerateOctave(image, sigma_0, S):
    """
    生成一组octave，即金字塔的一层
    :param image:输入图像
    :param sigma_0:
    :param S: 每一组octave可以提取特征点的层数
    :return: 一组octave
    """
    k = 2**(1/S)
    image_num_per_octave = S + 3
    one_octave = np.zeros([image.shape[0], image.shape[1], image_num_per_octave])
    one_octave[:, :, 0] =image
    # one_octave[:, :, 0] = MaxMinNorm(one_octave[:, :, 0])
    # cv.imshow("test", one_octave[:, :, 0])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    for i in range(1,image_num_per_octave):
        sigma = (k ** (i)) * sigma_0
        kernel_shape = int(round(6 * sigma) + 1)
        gaussian_kernel = GaussianKernel(sigma, kernel_shape)
        one_octave[:, :, i] = conv2d(image, gaussian_kernel)   # cv.GaussianBlur函数也可以实现
        # one_octave[:, :, i] = MaxMinNorm(one_octave[:, :, i])
        # cv.imshow("test", one_octave[:, :, i])
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    return one_octave

def GaussianPyramid(image, sigma_0, S):
    """
    生成高斯金字塔
    :param image: 输入图像
    :param sigma_0: sigma
    :param S: 每一组octave可以提取特征点的层数
    :return: 高斯金字塔
    """
    num_octave = int(round(np.log(min(image.shape)) / np.log(2) - 3))
    gaussian_pyramid = []

    # 生成第一组octave的第一层
    kernel_shape = int(round(6 * sigma_0) + 1)
    gaussian_kernel_0 = GaussianKernel(sigma_0, kernel_shape)
    gaussian_pyramid_0 = conv2d(image, gaussian_kernel_0)

    # 生成第一组octave
    first_octave = GenerateOctave(gaussian_pyramid_0, sigma_0, S)
    gaussian_pyramid.append(first_octave)

    # 生成其余octave
    temp_first_layer = DownSample(first_octave[:,:,-3])
    for i in range(1, num_octave):
        sigma = sigma_0 * (2 ** i)
        temp_octave = GenerateOctave(temp_first_layer, sigma, S)
        gaussian_pyramid.append(temp_octave)
        temp_first_layer = DownSample(temp_octave[:,:,-3])
    return gaussian_pyramid

def DoGPyramid(image, sigma_0, S):
    """
    生成DOG金字塔
    :param image:  输入图像
    :param sigma_0:
    :param S: 每一组octave可以提取特征点的层数
    :return: DOG金字塔
    """
    gaussian_pyramid = GaussianPyramid(image, sigma_0, S)
    image_num_per_octave = S + 2
    DOG_pyramid = []
    for octave in gaussian_pyramid:
        DOG_octave = np.zeros([octave.shape[0], octave.shape[1], image_num_per_octave])
        for i in range(image_num_per_octave):
            DOG_octave[:, :, i] = octave[:, :, i+1] - octave[:, :, i]
        DOG_pyramid.append(DOG_octave)

    return DOG_pyramid

# 1.2 DoG尺度空间极值点检测，确定初步候选点

# 循环先把解找到，返回坐标[octave, i, j, s]
def DetectExtremePoint(DOG_pyramid):

    """
    返回DoG尺度空间极值点坐标
    :param DOG_pyramid: DoG金字塔
    :return: DoG空间极值点坐标列表
    """
    num_layers = DOG_pyramid[0].shape[2]
    extreme_points = []
    octave_index = 0
    for octave in DOG_pyramid:
        L = octave.shape[0]
        H = octave.shape[1]
        temp_region = np.zeros([3,3,3])
        for s in range(1, num_layers-1):
            up_layer = octave[:, :, s-1]
            middle_layer = octave[:, :, s]
            down_layer = octave[:, :, s+1]
            for i in range(1, L-1):
                for j in range(1, H-1):
                    extreme_point_coordinate = np.asarray([octave_index, i, j, s])   # 记录当前中心点坐标
                    temp_region[:, :, 0] = up_layer[i - 1:i + 2, j - 1:j + 2]
                    temp_region[:, :, 1] = middle_layer[i - 1:i + 2, j - 1:j + 2]
                    temp_region[:, :, 2] = down_layer[i - 1:i + 2, j - 1:j + 2]
                    temp_coordinate = np.asarray([1,1,1])   # 指中心坐标
                    i_max, j_max, s_max = np.where(temp_region == np.max(temp_region))
                    # print(np.where(temp_region == np.max(temp_region)))
                    i_min, j_min, s_min = np.where(temp_region == np.min(temp_region))
                    temp_max_coordinate = np.hstack((i_max, j_max, s_max))
                    temp_min_coordinate = np.hstack((i_min, j_min, s_min))
                    # 如果中心坐标与极值点坐标相符，说明是极值点，需要保存
                    if (temp_max_coordinate == temp_coordinate).all() or (temp_min_coordinate == temp_coordinate).all():
                        extreme_points.append(extreme_point_coordinate)
        octave_index = octave_index + 1
    return extreme_points

"""************** step2：关键点的精确定位，同时去除不稳定的关键点 *****************"""

def ComputeGradient(DOG_pyramid, X):
    actove = DOG_pyramid[X[0]]
    i = X[1]; j = X[2]; s = X[3]
    D_dx = 0.5 * (actove[i, j+1, s] - actove[i, j-1, s])
    D_dy = 0.5 * (actove[i+1, j, s] - actove[i-1, j, s])
    D_ds = 0.5 * (actove[i, j, s+1] - actove[i, j, s-1])
    D_gradient = np.hstack((D_dx, D_dy, D_ds))
    return D_gradient

def ComputeHessian(DOG_pyramid, X):
    actove = DOG_pyramid[X[0]]
    i = X[1]; j = X[2]; s = X[3]
    D_dxx = actove[i, j+1, s] + actove[i, j-1, s] - 2*actove[i, j, s]
    D_dyy = actove[i+1, j, s] + actove[i-1, j, s] - 2*actove[i, j, s]
    D_dss = actove[i, j, s+1] + actove[i, j, s-1] - 2*actove[i, j, s]
    D_dxy = ((actove[i+1, j+1, s] + actove[i-1, j-1, s]) - (actove[i-1, j+1, s] + actove[i+1, j-1, s]))/4
    D_dxs = ((actove[i, j+1, s+1] + actove[i, j-1, s-1]) - (actove[i, j-1, s+1] + actove[i, j+1, s-1]))/4
    D_dys = ((actove[i+1, j, s+1] + actove[i-1, j, s-1]) - (actove[i-1, j, s+1] + actove[i+1, j, s-1]))/4
    D_hessian = np.asarray([[D_dxx, D_dxy, D_dxs],
                            [D_dxy, D_dyy, D_dys],
                            [D_dxs, D_dys, D_dss]])
    return D_hessian

def ComputeOffest(DOG_pyramid, X):
    D_gradient =ComputeGradient(DOG_pyramid, X)
    D_hessian = ComputeHessian(DOG_pyramid, X)
    X_offest = np.linalg.lstsq(D_hessian, D_gradient)[0]  # 最小二乘法解出偏移量
    return X_offest

# 2.1 对候选点进行精确定位
def DetectKeyPoint(DOG_pyramid):
    extreme_points = DetectExtremePoint(DOG_pyramid)
    key_points = []
    for point_coordinate in extreme_points:
        # point_coordinate = [octave, i, j, s]
        octave = point_coordinate[0]
        i = point_coordinate[1]
        j = point_coordinate[2]
        s = point_coordinate[3]

        num_i = DOG_pyramid[octave].shape[0]
        num_j = DOG_pyramid[octave].shape[1]
        num_s = DOG_pyramid[octave].shape[2]
        # 确定初步极值点的DOG值和梯度，为修正做准备
        D_0 = DOG_pyramid[octave][i, j, s]
        D_gradient = ComputeGradient(DOG_pyramid, point_coordinate)

        # 迭代N次对极值点的值和坐标进行修正
        N = 5
        for n in range(N):
            X_offest = ComputeOffest(DOG_pyramid, point_coordinate)
            if (abs(X_offest) > np.asarray([0.5, 0.5, 0.5])).any():
                point_coordinate[1:] = point_coordinate[1:]+np.round(X_offest)
                if point_coordinate[1] > 0 and point_coordinate[2] > 0 and point_coordinate[3] > 0 and \
                        point_coordinate[1] < num_i - 1 and point_coordinate[2] < num_j - 1 and point_coordinate[
                    3] < num_s - 1:
                    if n == N-1:
                        D_value = D_0 + 0.5 * np.dot(D_gradient, X_offest)
                        point = np.hstack((np.asarray(D_value), point_coordinate))
                        if RemoveUnstablePoint(DOG_pyramid, point):
                            key_points.append(point)  # 将修正好的点加入关键点列表
                else:
                    break
            else:
                point = np.hstack((np.asarray(D_0), point_coordinate))
                if RemoveUnstablePoint(DOG_pyramid, point):
                    key_points.append(point)  # 将不用修正的点加入关键点列表
                break

    key_points = np.asarray(key_points)
    return key_points

# 2.2 去除不稳定点（对比度低和边缘点）
def RemoveUnstablePoint(DOG_pyramid, key_point):
    # key_point = [value, octave, i, j, s]
    if abs(key_point[0]) < 0.3:
        return False
    actove = DOG_pyramid[int(key_point[1])]
    i = int(key_point[2])
    j = int(key_point[3])
    s = int(key_point[4])
    D_dxx = actove[i, j + 1, s] + actove[i, j - 1, s] - 2 * actove[i, j, s]
    D_dyy = actove[i + 1, j, s] + actove[i - 1, j, s] - 2 * actove[i, j, s]
    D_dxy = ((actove[i + 1, j + 1, s] + actove[i - 1, j - 1, s]) - (
                actove[i - 1, j + 1, s] + actove[i + 1, j - 1, s])) / 4
    tr = D_dxx + D_dyy
    det  = D_dxx * D_dyy - D_dxy * D_dxy
    if det <= 0:
        return True
    r = 10
    if (tr * tr)/det > (r + 1) ** 2/r:
        return False
    return True


# 2.3 将检测到的特征点显示在图像上
def DrawFeaturePoint(image, key_points):
    # key_point = [value, octave, i, j, s]
    for k in range(len(key_points)):
        key_point = key_points[k, :]
        octave = int(key_point[1])
        i = int(key_point[2]*pow(2, octave))
        j = int(key_point[3]*pow(2, octave))
        if i >= image.shape[1] or j >= image.shape[0]:
            continue
        cv.circle(image, center=(j,i), radius=2, color=(0, 255, 0))
    cv.imshow('feature_point', image)
    cv.waitKey()
    return image

# if __name__ == '__main__':
#     img = cv.imread('data/Lena.tif', 0)
#
#     S = 3        #S表示提取特征点的层数
#     sigma_0 = 1.6         #sigma为模糊尺度
#     DOG_pyramid = DoGPyramid(img, sigma_0, S)
#     key_points = DetectKeyPoint(DOG_pyramid)
#     DrawFeaturePoint(img, key_points)
#     m =2
