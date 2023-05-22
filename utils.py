import numpy as np


def generate_random_distance_matrix(n):
    random_matrix = np.random.rand(n, n)
    distance_matrix = random_matrix + random_matrix.T
    # Set zero to the diagonal
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix