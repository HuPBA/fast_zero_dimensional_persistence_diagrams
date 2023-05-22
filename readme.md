# Fast computation of zero dimensional Vietoris-Rips persistence diagrams.

Zero dimensional persistence diagrams computed using the Vietoris-Rips filtration for a point cloud $X$ and a symmetric function $d(x,y)\geq 0$ such that $d(x,x)=0$ can be computed as the multisets of points $\{(0,d):d\in\text{MST}(X,d)\}$, where $\text{MST}(X,d)$ is the multiset of edge weights in the minimum spanning tree of the clique graph $G=(X,E)$ with edge weights $d(x,y)$. In particular, these weights can be computed using the single-linkage clustering algorithm. The single-linkage algorithm has a complexity of $\mathcal{O}(n^2)$, lower than the usual matrix reduction algorithms to compute persistence diagrams. In this repo, we adapt the code of the [single linkage algorithm imlpemented by SciPy](https://github.com/scipy/scipy/blob/v1.10.1/scipy/cluster/_hierarchy.pyx) in such a way it also returns the indices of the distance matrix corresponding to each of the weights of the minimum spanning tree. This way, differentiating persistence diagrams using autograd algorithms is very simple: examples for PyTorch and Tensorflow frameworks are included in the ``examples`` subfolder. 

The file ``comparison_methods.py`` contains a code prepared to compare the performances of the main software packages for computing persistence diagrams, Gudhi and Ripser, the ``scipy`` implementation of the single linkage clustering algorithm, and our slightly modified implementation. For a basic test, the following table contains the execution time, in seconds, of each method for point clouds in $\mathbb R^2$ of 10, 100, 1000, and 10000 points, respectively.

|                                       |          10 |         100 |      1000 |     10000 |
|---------------------------------------|-------------|-------------|-----------|-----------|
| our_implementation | 0.000121872 | 0.000263299 | 0.017938  |   4.70886 |
| scipy_single_linkage                  | 0.000292492 | 0.000439857 | 0.0169241 |   4.50809 |
| ripser_parallel                       | 0.000550624 | 0.00186656  | 0.23026   |  33.534   |
| gudhi                                 | 0.000121593 | 0.00620066  | 1.21829   | 173.289   |

As you can see, differences between performances can be seen even for the computation of persistence diagrams of small point clouds with 100 points. 

## Installation

You can install the package directly by executing the command:

``pip install git+https://github.com/HuPBA/fast_zero_dimensional_persistence_diagrams``

## Usage

To use the package, you first need to import it:

``import zero_persistence_diagram``

Then, you can use it with the header

``zero_persistence_diagram.zero_persistence_diagram_by_single_linkage_algorithm(condensed_distance_matrix)
``

The only parameter the functions accepts is ``condensed_distance_matrix``. This is the condensed version of a full distance matrix. See ``scipy.spatial.distance.squareform``. 

The output of the function is a pair consisting of the deaths of the persistence diagram (weights of the MST) and a ``numpy`` list of condensed indices of the same size of the persistence diagram, respectively. The $i$-th condensed index corresponds to the position in the condensed distance matrix of the $i$-th point of the persistence diagram returned by the method. To obtain the index of the full distance matrix you can use the following function.

```
def get_indices_from_condensed_index(condensed_index, number_of_points):
   '''
   condensed_index: int -> A condensed index of the condensed indices array returned by the function zero_persistence_diagram_by_single_linkage_algorithm.
   number_of_points: int -> Number of points of the point cloud. Length of the full distance matrix. 
   '''
    b = 1 - (2 * number_of_points)
    i = int((-b - math.sqrt(b ** 2 - 8 * condensed_index)) // 2)
    j = condensed_index + i * (b + i + 2) // 2 + 1
    return i, j
```

To generate differentiable zero dimensional persistence diagrams, you can use the following pieces of code. For Tensorflow:

```
def compute_differentiable_persistence_diagram(point_cloud):
    # Compute the distance matrix. Note that the computation must be using a function
    # that is differentiable with respect to the point cloud.
    distance_matrix = compute_distance_matrix(point_cloud)
    # Compute the persistence diagram without backprop
    condensed_distance_matrix = scipy.spatial.distance.squareform(tf.stop_gradient(distance_matrix).numpy(),
                                                                  checks=False)
    pd, condensed_pairs = zero_persistence_diagram.zero_persistence_diagram_by_single_linkage_algorithm(
        condensed_distance_matrix)
    pairs = tf.constant([get_indices_from_condensed_index(condensed_index, point_cloud.shape[0]) for condensed_index in
                         condensed_pairs])
    # Filter the distance matrix to have the pairs we want
    pd_according_to_pairs = tf.gather_nd(distance_matrix, pairs)
    return pd_according_to_pairs
```

For PyTorch:

```
def compute_persistence_diagram(point_cloud):
    # Compute the distance matrix. Note that the computation must be using a function
    # that is differentiable with respect to the point cloud.
    distance_matrix = torch.cdist(point_cloud, point_cloud, p=2)
    # Compute the persistence diagram without backprop
    with torch.no_grad():
        condensed_distance_matrix = scipy.spatial.distance.squareform(distance_matrix.detach(), checks=False)
        pd, condensed_pairs = zero_persistence_diagram.zero_persistence_diagram_by_single_linkage_algorithm(
            condensed_distance_matrix)
    pairs = torch.tensor([get_indices_from_condensed_index(condensed_index, point_cloud.shape[0]) for condensed_index in
                          condensed_pairs])
    # Filter the distance matrix to have the pairs we want
    pd_according_to_pairs = distance_matrix[pairs[:, 0], pairs[:, 1]]
    return pd_according_to_pairs
```

## Further details

This project is mantained by [Rubén Ballester Bautista](https://rubenbb.com) ([@rballeba](https://github.com/rballeba)). We do not plan to update the repository unless there are critical bugs on the implementation. The idea of this repository is to allow practitioners to compute zero persistence diagrams in a fast way, with the possibility of using the differentiability to impose certain *shapes* to the point clouds. This could be used, for example, to regularise neural networks or machine learning algorithms.