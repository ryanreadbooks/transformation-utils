import numpy as np
import math


def angles_to_rot_mat(rx: float, ry: float, rz: float, rot_type='abg'):
    """
    绕三个轴的旋转角度化成一个旋转矩阵
    :param rx: 绕x轴的旋转量，单位(rad)
    :param ry: 绕y轴的旋转量，单位(rad)
    :param rz: 绕z轴的旋转量，单位(rad)
    :param rot_type: 设置三个旋转角度化成旋转矩阵的形式，可选'abg'和'gba'。
                    其中abg的三次旋转都是绕定轴的；gba的三次旋转是绕动轴的。
    :return: 对应的旋转矩阵
    """

    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(rx), -np.sin(rx)],
                           [0, np.sin(rx), np.cos(rx)]])

    rotation_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                           [0, 1, 0],
                           [-np.sin(ry), 0, np.cos(ry)]])

    rotation_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                           [np.sin(rz), np.cos(rz), 0],
                           [0, 0, 1]])

    if rot_type == 'abg':
        rot = (rotation_z @ rotation_y) @ rotation_x  # 'abg'
    else:
        rot = (rotation_x @ rotation_y) @ rotation_z  # 'gba'

    return rot


"""
以下三个函数来自https://github.com/pyni/matrix_from_twovectors/blob/master/rotation_calculation.py
"""
def angle2rotm_4x4(angle, axis, point=None):
    """
    通过轴角计算出旋转矩阵
    :param angle: 旋转的角度，单位rad
    :param axis: 旋转轴，np.ndarray
    """
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis/np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[ 0.0,     -axis[2],  axis[1]],
                      [ axis[2], 0.0,      -axis[0]],
                      [-axis[1], axis[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M
    

def angle2rotm_3x3(angle: float, axis, point=None):
    """
    通过轴角计算出旋转矩阵
    :param angle: 旋转的角度，单位rad
    :param axis: 旋转轴，np.ndarray
    """
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis/np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[ 0.0,     -axis[2],  axis[1]],
                      [ axis[2], 0.0,      -axis[0]],
                      [-axis[1], axis[0],  0.0]])
 
    if point is not None:
        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        R = point - np.dot(R, point)
    return R 


def rotation_calculation_with3x3matrix(vectorBefore, vectorAfter):
    """
    计算两个向量之间的旋转矩阵。从vectorBefore到vectorAfter的旋转矩阵
    """
    vectorBefore = np.array(vectorBefore)
    vectorAfter = np.array(vectorAfter)
    rotation_axis = np.cross(vectorBefore, vectorAfter)  
    rotationangle = math.acos(np.dot(vectorBefore, vectorAfter) / (np.linalg.norm(vectorBefore) * np.linalg.norm(vectorAfter)))
    matrix = angle2rotm_3x3(rotationangle, rotation_axis)
    return matrix


def rot_mat_diff(r1, r2):
    """
    计算两个旋转矩阵之间的角度差(单位rad)
    """
    assert r1.shape == (3, 3) and r2.shape == (3, 3), 'rotation matrix must be 3x3'
    tr = np.trace(r1 @ r2.T)
    return np.arccos(0.5 * (tr - 1))
