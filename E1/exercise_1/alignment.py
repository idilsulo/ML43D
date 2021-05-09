""" Procrustes Aligment for point clouds """
import numpy as np
from pathlib import Path


def procrustes_align(pc_x, pc_y):
    """
    calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
    :param pc_x: Nx3 input point cloud
    :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
    :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
    """
    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    # TODO: Your implementation starts here ###############
    # 1. get centered pc_x and centered pc_y
    # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
    # 3. estimate rotation
    # 4. estimate translation
    # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)

    X = np.transpose(pc_x)
    Y = np.transpose(pc_y)

    # Get the mean of all samples for each point cloud
    X_mean = np.mean(X, axis=1).reshape(3,1)
    Y_mean = np.mean(Y, axis=1).reshape(3,1)
    
    # Remove translation by mean-centering
    X = X - X_mean
    Y = Y - Y_mean

    # Compute SVD 
    X_Y_t = np.matmul(X, Y.transpose())
    U, S, V_t = np.linalg.svd(X_Y_t)

    epsilon = 1e-8 # Check equality based on the provided error rate, epsilon
    if np.abs(np.linalg.det(U) * np.linalg.det(V_t.transpose()) - 1) <= epsilon:
        S = np.eye(3)
    else:
        S = np.diagflat([1,1,-1])

    # Estimate the rotation matrix based on SVD results
    R = np.matmul(np.matmul(U, S), V_t).transpose()

    # Estimate the translation
    # Recover the translation from the version where translation is removed: R @ X_mean
    t = (Y_mean - np.matmul(R, X_mean)).reshape(3)

    # TODO: Your implementation ends here ###############

    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
    print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())

    return R, t


def load_correspondences():
    """
    loads correspondences between meshes from disk
    """

    load_obj_as_np = lambda path: np.array(list(map(lambda x: list(map(float, x.split(' ')[1:4])), path.read_text().splitlines())))
    path_x = (Path(__file__).parent / "resources" / "points_input.obj").absolute()
    path_y = (Path(__file__).parent / "resources" / "points_target.obj").absolute()
    return load_obj_as_np(path_x), load_obj_as_np(path_y)
