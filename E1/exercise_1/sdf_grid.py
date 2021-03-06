"""Creating an SDF grid"""
import numpy as np


def sdf_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with value 0 outside the shape and 1 inside.
    """

    # ###############
    # TODO: Implement

    grid = np.zeros((resolution, resolution, resolution))
    step = 1.0 / (resolution - 1)
    coord = np.arange(-0.5, 0.5+step, step)
    
    # Uses scalar input for sdf function
    # for i in range(resolution):
    #     for j in range(resolution):
    #         for k in range(resolution):
    #             grid[i][j][k] = sdf_function(coord[i],coord[j],coord[k])


    # Use 1D numpy arrays instead
    for i in range(resolution):
        for j in range(resolution):
            x = np.array([coord[i]] * resolution)
            y = np.array([coord[j]] * resolution)
            z = coord.copy()
            
            sdf = sdf_function(x, y, z)
            grid[i][j] = sdf.copy()


    return grid
    # ###############
