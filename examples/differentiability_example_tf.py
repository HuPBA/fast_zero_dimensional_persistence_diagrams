import math

import scipy
import tensorflow as tf
import zero_persistence_diagram
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_indices_from_condensed_index(condensed_index, number_of_points):
    b = 1 - (2 * number_of_points)
    i = int((-b - math.sqrt(b ** 2 - 8 * condensed_index)) // 2)
    j = condensed_index + i * (b + i + 2) // 2 + 1
    return i, j

def compute_differentiable_persistence_diagram(point_cloud):
    # Compute euclidean distance matrix for points in the point cloud
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

def compute_distance_matrix(point_cloud):
    number_of_points = point_cloud.shape[0]
    number_of_dimensions = point_cloud.shape[1]
    t1 = tf.reshape(point_cloud, (1, number_of_points, number_of_dimensions))
    t2 = tf.reshape(point_cloud, (number_of_points, 1, number_of_dimensions))
    return tf.norm(t1 - t2, ord='euclidean', axis=2, )


def loss_function(point_cloud):
    pd_according_to_pairs = compute_differentiable_persistence_diagram(point_cloud)
    # Compute the loss. In this case, we minimise the total persistence of the persistence
    # diagram.
    return tf.math.reduce_sum(pd_according_to_pairs)


# Generate a gif of the optimization process
def generate_gif(point_clouds):
    fig = plt.figure()
    ims = []
    for point_cloud in point_clouds:
        im = plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c='b')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save('./animation_tf.gif', writer='imagemagick')


def optimize_point_cloud(number_of_iterations, number_of_points):
    point_clouds = []
    point_cloud = tf.Variable(tf.random.uniform([number_of_points, 2], minval=0, maxval=3))
    point_cloud.requires_grad = True
    point_clouds.append(tf.stop_gradient(point_cloud).numpy())
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for i in range(number_of_iterations):
        with tf.GradientTape() as tape:
            loss = loss_function(point_cloud)
        gradients = tape.gradient(loss, point_cloud)
        optimizer.apply_gradients(zip([gradients], [point_cloud]))
        point_clouds.append(tf.stop_gradient(point_cloud).numpy())
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{number_of_iterations}, Loss: {loss.numpy()}")
    return point_clouds


def main():
    point_clouds = optimize_point_cloud(400, 200)
    generate_gif(point_clouds)


if __name__ == '__main__':
    main()
