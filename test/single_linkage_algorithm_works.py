# test if persistence diagrams obtained with ripser and with the single linkage algorithm are equal
from random import random

import numpy as np
import scipy
import zero_persistence_diagram
from gph import ripser_parallel

from utils import generate_random_distance_matrix


def test_persistence_diagrams_are_correct(number_of_distance_matrices, size_of_distance_matrices=100):
    for _ in range(number_of_distance_matrices):
        distance_matrix = generate_random_distance_matrix(size_of_distance_matrices)
        condensed_distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
        ripser_pd = ripser_parallel(distance_matrix, metric='precomputed', maxdim=0)['dgms'][0][:, 1]
        pd, pairs = zero_persistence_diagram.zero_persistence_diagram_by_single_linkage_algorithm(
            condensed_distance_matrix)
        pd_according_to_pairs = [condensed_distance_matrix[pair] for pair in pairs]
        ripser_pd = sorted(ripser_pd)[:-1]  # Remove last infinity point
        pd_according_to_pairs = sorted(pd_according_to_pairs)
        pd = sorted(pd)
        assert np.allclose(ripser_pd, pd)
        assert np.allclose(ripser_pd, pd_according_to_pairs)
        assert np.allclose(pd_according_to_pairs, pd)


if __name__ == '__main__':
    test_persistence_diagrams_are_correct(1000)
