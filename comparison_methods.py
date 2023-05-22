import timeit
from functools import partial

import gudhi
import scipy.cluster.hierarchy
import tabulate
import zero_persistence_diagram
from gph import ripser_parallel

from utils import generate_random_distance_matrix


# Methods for computing persistence diagram


def persistence_diagram_scipy(distance_matrix):
    # Convert distance matrix to condensed distance matrix
    condensed_distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
    return scipy.cluster.hierarchy.linkage(condensed_distance_matrix, method='single')[:, 2]


def persistence_diagram_ripser_ph(distance_matrix):
    return ripser_parallel(distance_matrix, metric='precomputed', maxdim=0)[0][:, 1]


def persistence_diagram_gudhi(distance_matrix):
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    persistence_diagrams = simplex_tree.persistence(
        homology_coeff_field=2)
    return persistence_diagrams


def persistence_diagram_single_linkage(distance_matrix):
    condensed_distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
    return zero_persistence_diagram.zero_persistence_diagram_by_single_linkage_algorithm(condensed_distance_matrix)[0]


# Benchmark details

benchmark_sizes = [10, 100, 1000, 10000]

# Methods for computing persistence diagram

methods = {
    'single_linkage_with_persistence_pairs': persistence_diagram_single_linkage,
    'scipy_single_linkage': persistence_diagram_scipy,
    'ripser_parallel': partial(ripser_parallel, metric='precomputed', maxdim=0),
    'gudhi': persistence_diagram_gudhi,
}


def tabulate_results(benchmark_results, method_names, benchmark_sizes):
    benchmark_table = []
    for method_name in method_names:
        benchmark_table.append(benchmark_results[method_name])
    return tabulate.tabulate(benchmark_table, headers=benchmark_sizes, tablefmt='github', showindex=method_names)


def benchmark_methods_for_computing_persistence_diagram(number_of_experiments=1):
    method_names = methods.keys()
    methods_and_times = {}
    for n in benchmark_sizes:
        distance_matrix = generate_random_distance_matrix(n)
        for method_name in method_names:
            method = methods[method_name]
            print(f'Benchmarking {method_name} for n={n}')
            function_call = partial(method, distance_matrix)
            elapsed_time_total = timeit.timeit(function_call, number=5)
            elapsed_time = elapsed_time_total / number_of_experiments
            if method_name not in methods_and_times:
                methods_and_times[method_name] = [elapsed_time]
            else:
                methods_and_times[method_name].append(elapsed_time)
    print(tabulate_results(methods_and_times, method_names, benchmark_sizes))


if __name__ == '__main__':
    benchmark_methods_for_computing_persistence_diagram()
