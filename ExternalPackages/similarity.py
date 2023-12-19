import numpy as np


def euclidianDist(image1, image2):
    # Flatten 3d image arrays
    i1 = np.matrix.flatten(image1)
    i2 = np.matrix.flatten(image2)

    return np.linalg.norm(i1 - i2)

