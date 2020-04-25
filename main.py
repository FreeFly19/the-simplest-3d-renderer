import math

import cv2
import numpy as np

from renderer_functions import *

camera1_pos = np.array([0, 0, 0])
camera1_rotations = np.array([0, 0, 0])


camera_f = 0.25
output_img_size = np.array([320, 180])

camera1 = np.array([
    [output_img_size[0] * camera_f, 0, output_img_size[1] / 2, 0],
    [0, output_img_size[0] * camera_f, output_img_size[0] / 2, 0],
    [0, 0, 1, 0],
], dtype=np.float64)

camera1 = camera1 @ generate_rotation_z(-math.pi / 2)

camera1 = camera1 @ generate_rotation_x(math.pi / 180 * camera1_rotations[0])
camera1 = camera1 @ generate_rotation_y(math.pi / 180 * camera1_rotations[1])
camera1 = camera1 @ generate_rotation_z(math.pi / 180 * camera1_rotations[2])


###

cube_pos = np.array([0, 0, -15])
cube_size = np.array([3, 4, 3])
points_to_render = cube_points(cube_pos, cube_size)

cube2_pos = np.array([8, 3, -10])
cube2_size = np.array([4, 5, 6])
points_to_render = np.concatenate([points_to_render, cube_points(cube2_pos, cube2_size)])


###

output_img = np.ones((output_img_size[1], output_img_size[0]), dtype=np.uint8) * 255

draw_3d_distance_points(output_img, points_to_render, camera1, max_distance=1000)

output_img = cv2.resize(output_img, dsize=(1280, 720), interpolation=cv2.INTER_NEAREST)
cv2.imshow('output', output_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
