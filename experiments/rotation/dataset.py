from torch.utils.data import Dataset
import torch
import numpy as np
import math
from coordinate_transformation import spherical_to_euclidean, euclidean_to_spherical, get_spherical_rotation, \
    apply_euclidean_rotation


class RotationDataset(Dataset):
    """Dataloder for the Rotation datasets"""

    def __init__(self, dim, n, seed=1683):
        np.random.seed(seed)
        points_list = []
        angles_list = []
        points_rotated_list = []
        for p in range(n):
            # sample random point on unit hypersphere in first quarter
            point1 = np.random.normal(size=dim)
            point1 /= np.linalg.norm(point1)

            # point2 = np.random.normal(size=dim)
            # point2 /= np.linalg.norm(point2)

            # point2sph = euclidean_to_spherical(point1) + np.array([math.pi/4, math.pi/4])
            # point2 = spherical_to_euclidean(point2sph)

            # rotation angles are difference of spherical coordinates
            # rotation_angles = get_spherical_rotation(euclidean_to_spherical(point1), euclidean_to_spherical(point2))

            rotation_angles = np.random.uniform(-math.pi, math.pi, size=dim-1)
            point1_rotated = apply_euclidean_rotation(point1, rotation_angles)
            if seed > 0 and False:
                print("-----------------------------------")
                print("Original point euclidean coordinates")
                print(point1.round(8))
                print("Original point spherical coordinates")
                print(euclidean_to_spherical(point1).round(8))
                print("Rotated point euclidean coordinates")
                print(point1_rotated.round(8))
                print("Rotated point spherical coordinates")
                print(euclidean_to_spherical(point1_rotated).round(8))
                print("Rotation angles")
                print(rotation_angles.round(8))
                print("Diff spherical coordinates")
                print(get_spherical_rotation(euclidean_to_spherical(point1), euclidean_to_spherical(point1_rotated)).round(8))
                # raise Exception("exc")
            points_list.append(point1)
            angles_list.append(rotation_angles)
            points_rotated_list.append(point1_rotated)

        self.points = torch.FloatTensor(points_list)
        self.angles = torch.FloatTensor(angles_list)
        self.points_rotated = torch.FloatTensor(points_rotated_list)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx], self.angles[idx], self.points_rotated[idx]



class RotationDatasetOLD(Dataset):
    """Dataloder for the Rotation datasets"""

    def __init__(self, dim, n, seed=1683):
        np.random.seed(seed)
        points_list = []
        angles_list = []
        points_rotated_list = []
        for _ in range(n):
            # sample two random points on unit hypersphere
            point1 = np.random.normal(size=dim)
            point1 /= np.linalg.norm(point1)
            # point2 = np.random.normal(size=dim)
            # point2 /= np.linalg.norm(point2)
            # rotation angles are difference of spherical coordinates
            # rotation_angles = get_spherical_rotation(euclidean_to_spherical(point1), euclidean_to_spherical(point2))
            rotation_angles = np.random.uniform(-math.pi, math.pi, size=dim-1)
            point1_rotated = apply_euclidean_rotation(point1, rotation_angles)
            points_list.append(point1)
            angles_list.append(rotation_angles)
            points_rotated_list.append(point1_rotated)

        self.points = torch.FloatTensor(points_list)
        self.angles = torch.FloatTensor(angles_list)
        self.points_rotated = torch.FloatTensor(points_rotated_list)

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx], self.angles[idx], self.points_rotated[idx]
