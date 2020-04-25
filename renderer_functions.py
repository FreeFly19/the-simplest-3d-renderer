import numpy as np
from math import *


def generate_rotation_x(angle, point=False):
    angle = -angle
    m = np.array([
        [1, 0, 0, 0],
        [0, cos(angle), -sin(angle), 0],
        [0, sin(angle), cos(angle), 0],
        [0, 0, 0, 1]
    ])
    if point:
        return m[:3, :3]
    return m


def generate_rotation_y(angle, point=False):
    angle = -angle
    m = np.array([
        [cos(angle), 0, sin(angle), 0],
        [0, 1, 0, 0],
        [-sin(angle), 0, cos(angle), 0],
        [0, 0, 0, 1]
    ])
    if point:
        return m[:3, :3]
    return m


def generate_rotation_z(angle, point=False):
    angle = -angle
    m = np.array([
        [cos(angle), -sin(angle), 0, 0],
        [sin(angle), cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    if point:
        return m[:3, :3]
    return m


def project_point(matrix, point):
    point_extended = np.append(point, [1])
    projected_point = matrix @ point_extended
    projected_point = projected_point / projected_point[2]
    return projected_point[:2]


def get_camera_pos(camera):
    return -np.linalg.inv(camera[:, :3]) @ camera[:, 3]


def translate_point(point, translation_vector):
    return np.array([
        [1, 0, 0, translation_vector[0]],
        [0, 1, 0, translation_vector[1]],
        [0, 0, 1, translation_vector[2]]
    ], dtype=float) @ point


def cube_points(pos, size):
    cube_points = np.array([
        [-size[0], size[1], -size[2]],
        [size[0], size[1], -size[2]],
        [size[0], size[1], size[2]],
        [-size[0], size[1], size[2]],
        [-size[0], -size[1], -size[2]],
        [size[0], -size[1], -size[2]],
        [size[0], -size[1], size[2]],
        [-size[0], -size[1], size[2]],
    ]) / 2

    cube_points[:, 0] = cube_points[:, 0] + pos[0]
    cube_points[:, 1] = cube_points[:, 1] + pos[1]
    cube_points[:, 2] = cube_points[:, 2] + pos[2]

    return cube_points


def draw_projected_point(img, projected_point, color):
    img_point = projected_point.astype(int)

    try:
        img[(-img_point[0], -img_point[1])] = color
    except IndexError:
        pass


def draw_3d_distance_points(img, points, camera, max_distance=300):
    camera_pos = get_camera_pos(camera)
    for p in points:
        distance = np.sum(np.power(camera_pos - p, 2))
        color = min(int(255 * distance / max_distance), 255)
        projected_point = project_point(camera, p)
        draw_projected_point(img, projected_point, color)
