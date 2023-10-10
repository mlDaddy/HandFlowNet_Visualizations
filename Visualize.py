import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from annot import read_obj_folder
from annot import read_npz_floder
from annot import read_MV_npz_floder, read_SV_npz_floder
import cv2
import numpy as np

def load_pnd_depth(depth_img_path, seg_mask):
    # Load the PNG depth image
    a = 240
    b = -140
    depth_image = cv2.resize(cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)[a:b, a:b], (640, 480))
    seg_mask = cv2.resize(seg_mask[a:b, a:b], (640, 480))
    depth_image2 = depth_image*seg_mask

    # Apply a colormap to visualize depth in color
    colored_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255/np.max(depth_image)), cv2.COLORMAP_JET)
    colored_depth2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image2, alpha=255/np.max(depth_image2)), cv2.COLORMAP_JET)

    # Display the original depth image and the colored depth image
    cv2.imshow('Original Depth Image', depth_image)
    cv2.imshow('Colored Depth Image', colored_depth)
    cv2.imshow('Hand Mask', seg_mask)
    cv2.imshow('Hand Mask 2', colored_depth2)

    cv2.imwrite('Original Depth Image.png', depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite('Colored Depth Image.png', colored_depth, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite('Hand Mask.png', seg_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite('Hand Mask 2.png', colored_depth2, [cv2.IMWRITE_PNG_COMPRESSION, 0])


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_vectors_3d(initial_points, terminal_points):
    """
    Plot 3D vectors with specified characteristics.

    Parameters:
    - initial_points (numpy.ndarray): Array of initial points for the vectors.
    - terminal_points (numpy.ndarray): Array of terminal points for the vectors.
    """

    # Calculate vector magnitudes
    vectors = terminal_points - initial_points
    magnitudes = np.linalg.norm(vectors, axis=1)

    # Normalize magnitudes to the [0, 1] range
    normalized_magnitudes = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes))

    # Create a color map ranging from blue to red
    # colors = plt.cm.RdBu(normalized_magnitudes)
    colors = plt.cm.brg(normalized_magnitudes)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot initial points (balls)
    ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2], c=colors, s=5, depthshade=False)
    # root_joint = 0
    # ax.scatter(initial_points[root_joint, 0], initial_points[root_joint, 1], initial_points[root_joint, 2], c=[0, 1, 0, 1], s=100, depthshade=False)

    # Plot vectors as arrows
    for i in range(len(initial_points)):
        ax.quiver(initial_points[i, 0], initial_points[i, 1], initial_points[i, 2],
                  vectors[i, 0], vectors[i, 1], vectors[i, 2],
                  color=colors[i], arrow_length_ratio=0.3)
        
    # Hide axes
    ax.axis('off')

    # Add a colorbar
    # cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdBu), ax=ax)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.brg), ax=ax)
    cbar.set_label('Normalized Vector Magnitude')

    # plt.legend()
    plt.show()


def plot_points_3d(points, colors=None, invert=False):
    
    if not isinstance(colors, np.ndarray):
        colors = [0, 0, 0, 1]

    if invert:
        points[:, 0] *= -1
    
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot initial points (balls)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1, depthshade=False)
        
    # Hide axes
    ax.axis('off')

    # Add a colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdBu), ax=ax)
    cbar.set_label('Normalized Vector Magnitude')

    # plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage
    
    hand_shape = read_obj_folder('Annotations', 72)
    # hand_shape[:, :, 0] *= -1
    hand_joints, seg_mask = read_npz_floder('joints', 72)

    frame_no = 20
    dt = 10
    initial_points = hand_shape[frame_no]
    terminal_points = hand_shape[frame_no + dt]
    sceneflow = hand_shape[frame_no + dt] - hand_shape[frame_no]

    plot_vectors_3d(initial_points, terminal_points)

    fdata = read_MV_npz_floder('HandsONly/20200908_144908', 72)
    # plot_points_3d(fdata[frame_no]['points_1'], fdata[frame_no]['colors_1'])
    # plot_points_3d(fdata[frame_no + dt]['points_1'], fdata[frame_no + dt]['colors_1'], invert=True)
    # plot_points_3d(fdata[frame_no]['points_1'])

    fdata = read_SV_npz_floder('DexYcb3d/20200908_144908/932122061900', 72)
    # plot_points_3d(fdata[frame_no]['hand_pcd_points'], fdata[frame_no]['hand_pcd_colors'])
    # plot_points_3d(fdata[frame_no + dt]['hand_pcd_points'], fdata[frame_no + dt]['hand_pcd_colors'], invert=True)
    # plot_points_3d(fdata[frame_no]['hand_pcd_points'])

    fdata = read_SV_npz_floder('DexYcb3d/20200908_144908/839512060362', 72)
    # plot_points_3d(fdata[frame_no]['hand_pcd_points'], fdata[frame_no]['hand_pcd_colors'])
    # plot_points_3d(fdata[frame_no + dt]['hand_pcd_points'], fdata[frame_no + dt]['hand_pcd_colors'], invert=True)
    points = fdata[frame_no]['hand_pcd_points']
    # points[:, 0] *= -1
    plot_points_3d(points)

    fdata = read_SV_npz_floder('DexYcb3d/20200908_144908/840412060917', 72)
    # plot_points_3d(fdata[frame_no]['hand_pcd_points'], fdata[frame_no]['hand_pcd_colors'])
    # plot_points_3d(fdata[frame_no + dt]['hand_pcd_points'], fdata[frame_no + dt]['hand_pcd_colors'], invert=True)
    points = fdata[frame_no]['hand_pcd_points']
    # points[:, 0] *= -1
    plot_points_3d(points)

    # load_pnd_depth('depth/aligned_depth_to_color_000020.png', seg_mask=seg_mask[frame_no])