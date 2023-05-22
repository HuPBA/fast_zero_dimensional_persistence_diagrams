import math

import numpy as np
import scipy
import torch
import zero_persistence_diagram
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_indices_from_condensed_index(condensed_index, number_of_points):
    b = 1 - (2 * number_of_points)
    i = int((-b - math.sqrt(b ** 2 - 8 * condensed_index)) // 2)
    j = condensed_index + i * (b + i + 2) // 2 + 1
    return i, j

def compute_persistence_diagram(point_cloud):
    # Compute euclidean distance matrix for points in the point cloud
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

def loss_function(point_cloud):
    pd_according_to_pairs = compute_persistence_diagram(point_cloud)
    # Compute the loss. In this case, we minimise the total persistence of the 
    # persistence diagram.
    return torch.sum(pd_according_to_pairs)


# Generate a gif of the optimization process
def generate_gif(point_clouds):
    fig = plt.figure()
    ims = []
    for point_cloud in point_clouds:
        im = plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c='r')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save('./animation_torch.gif', writer='imagemagick')


def optimize_point_cloud(number_of_iterations, number_of_points):
    point_clouds = []
    point_cloud = torch.rand(number_of_points, 2, requires_grad=False)*3
    point_cloud.requires_grad = True
    point_clouds.append(np.copy(point_cloud.detach().numpy()))
    optimizer = torch.optim.Adam([point_cloud], lr=0.01)
    for i in range(number_of_iterations):
        loss = loss_function(point_cloud)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        point_clouds.append(np.copy(point_cloud.detach().numpy()))
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{number_of_iterations}, Loss: {loss.item()}")
    return point_clouds


def main():
    point_clouds = optimize_point_cloud(400, 200)
    generate_gif(point_clouds)


if __name__ == '__main__':
    main()
