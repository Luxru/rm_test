import cv2
import numpy as np

# 2D图像坐标
image_points = np.array([
    (100, 150),
    (300, 150),
    (300, 300),
    (100, 300)
], dtype=np.float32)

# 3D物体坐标（转换为米）
object_points = np.array([
    (0, 0, 0),
    (0.1, 0, 0),
    (0.1, 0.1, 0),
    (0, 0.1, 0)
], dtype=np.float32)

# 摄像头参数
focal_length = 500
center = (image_points[2] + image_points[0]) / 2

camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# 使用PNP位姿解算
success, rotation_vec, translation_vec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

# 旋转向量转换为旋转矩阵
rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

print("旋转矩阵:")
print(rotation_matrix)
print("\n平移向量:")
print(translation_vec)
