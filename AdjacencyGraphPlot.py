import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_graph(adj_matrix, node_size=100, edge_width=1.0, display_labels=True, label_distance=0.02, node_positions=None):
    """
    Plot a 3D graph from an adjacency matrix.

    Parameters:
    adj_matrix (numpy.ndarray): Adjacency matrix of the graph.
    node_size (int): Size of the nodes.
    edge_width (float): Width of the edges.
    display_labels (bool): True to display node labels, False to display node numbers.
    label_distance (float): Distance of the labels from the nodes.
    node_positions (numpy.ndarray): Node positions in 3D space as a numpy array (N x 3).

    Returns:
    None
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_matrix(adj_matrix)

    # Generate 3D positions for nodes
    if node_positions is None:
        pos = nx.spring_layout(G, dim=3)
    else:
        pos = {i: node_positions[i] for i in range(len(node_positions))}

    # Plot the 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for node in G.nodes():
        x, y, z = pos[node]
        ax.scatter(x, y, z, s=node_size)  # Adjust node size here

        if display_labels:
            ax.text(x + label_distance, y + label_distance, z + label_distance, str(node), color='black')  # Display node number

    for edge in G.edges():
        u, v = edge
        x1, y1, z1 = pos[u]
        x2, y2, z2 = pos[v]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='black', linewidth=edge_width)

    # Hide grid, axis, and legend
    ax.grid(False)
    ax.axis('off')
    ax.legend().set_visible(False)

    plt.show()

# Sample adjacency matrix (replace with your own)
adj_matrix = np.array([
    #[0,1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],#0
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#1
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#2
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],#3
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#4
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#5
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],#6
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#7
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#8
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],#9
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],#10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],#11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],#12
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],#13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],#14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],#15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],#16
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#17
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],#19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],#20
    ])

# Sample node positions (replace with your own)
node_positions = np.array([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6],
                           [0.7, 0.8, 0.9],
                           [0.2, 0.5, 0.8]])

# Adjust node_size, edge_width, display_labels, label_distance, and node_positions as needed
plot_3d_graph(adj_matrix, node_size=200, edge_width=2.0, display_labels=True, label_distance=0.05, node_positions=None)

