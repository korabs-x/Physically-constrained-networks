import math
import numpy as np
from random import shuffle

"""
def get_rotation_matrix(angles):
    dim = len(angles) + 1
    # one rotation matrix for each angle
    rotation_matrices = [np.identity(dim) for _ in range(dim)]
    for axis, alpha in enumerate(angles):
        # rotate on plane spanned by axes a and b
        a = axis
        b = axis + 1
        mat_idx = len(rotation_matrices) - axis - 1
        rotation_matrices[mat_idx][a][a] = math.cos(alpha)
        rotation_matrices[mat_idx][a][b] = - math.sin(alpha)
        rotation_matrices[mat_idx][b][a] = math.sin(alpha)
        rotation_matrices[mat_idx][b][b] = math.cos(alpha)
    rotind = list(range(len(angles)))
    shuffle(rotind)
    print(rotind)
    return np.linalg.multi_dot(np.array(rotation_matrices)[rotind])
"""
def get_rotation_matrix(angles):
    dim = len(angles) + 1
    # one rotation matrix for each angle
    rotation_matrices = [np.identity(dim) for _ in range(dim)]
    for axis, alpha in enumerate(angles):
        # rotate on plane spanned by axes a and b
        a = 0
        b = axis + 1
        mat_idx = len(rotation_matrices) - axis - 1
        rotation_matrices[mat_idx][a][a] = math.cos(alpha)
        rotation_matrices[mat_idx][a][b] = - math.sin(alpha)
        rotation_matrices[mat_idx][b][a] = math.sin(alpha)
        rotation_matrices[mat_idx][b][b] = math.cos(alpha)
    # rotind = list(range(len(angles)))
    # shuffle(rotind)
    # print(rotind)
    # return np.linalg.multi_dot(np.array(rotation_matrices)[rotind])
    return np.linalg.multi_dot(rotation_matrices)


def apply_euclidean_rotation(point, angles):
    return np.dot(get_rotation_matrix(angles), point)


def apply_spherical_rotation(spherical_coordinates, angles):
    return spherical_coordinates + angles


def spherical_to_euclidean(spherical_coordinates):
    euclidean_coordinates = np.zeros(spherical_coordinates.shape[0] + 1)
    sin_product = 1
    for i, alpha in enumerate(spherical_coordinates):
        euclidean_coordinates[i] = sin_product * math.cos(alpha)
        sin_product *= math.sin(alpha)
    euclidean_coordinates[-1] = sin_product
    return euclidean_coordinates


def euclidean_to_spherical(euclidean_coordinates):
    spherical_coordinates = np.zeros(euclidean_coordinates.shape[0] - 1)
    square_sum = euclidean_coordinates[-1] ** 2 + euclidean_coordinates[-2] ** 2
    spherical_coordinates[-1] = math.acos(0 if square_sum == 0 else euclidean_coordinates[-2] / math.sqrt(square_sum))
    if euclidean_coordinates[-1] < 0:
        spherical_coordinates[-1] = 2 * math.pi - spherical_coordinates[-1]
    for i in reversed(range(0, euclidean_coordinates.shape[0] - 2)):
        square_sum += euclidean_coordinates[i] ** 2
        spherical_coordinates[i] = math.acos(min(1, max(-1, 0 if square_sum == 0 else euclidean_coordinates[i] / square_sum)))
    return spherical_coordinates


def get_spherical_rotation(sph1, sph2):
    rotation = sph2 - sph1
    for i in range(len(rotation)):
        if rotation[i] <= - math.pi:
            rotation[i] += 2 * math.pi
        elif rotation[i] > math.pi:
            rotation[i] -= 2 * math.pi
    return rotation

