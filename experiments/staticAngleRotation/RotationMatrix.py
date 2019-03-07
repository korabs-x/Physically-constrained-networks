import numpy as np
import math

alpha = math.pi / 4

def get_matrix(a=alpha):
    cosa = math.cos(a)
    sina = math.sin(a)
    return np.array([[cosa, -sina], [sina, cosa]])

# applys rotation of angle alpha to the point x, y around origin
def apply_rotation(x, y, a=alpha):
    return get_matrix(a).dot([x, y])