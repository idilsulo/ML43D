"""Definitions for Signed Distance Fields"""
import numpy as np


def signed_distance_sphere(x, y, z, r, x_0, y_0, z_0):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a sphere of radius r, centered at (x_0, y_0, z_0)
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :param r: radius of the sphere
    :param x_0: x coordinate of the center of the sphere
    :param y_0: y coordinate of the center of the sphere
    :param z_0: z coordinate of the center of the sphere
    :return: signed distance from the surface of the sphere
    """

    # ###############
    # TODO: Implement

    dist_to_surface = np.sqrt((x-x_0)**2 + (y-y_0)**2 + (z-z_0)**2)-r
    return dist_to_surface
    # ###############


def signed_distance_torus(x, y, z, R, r, x_0, y_0, z_0):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a torus of minor radius r and major radius R, centered at (x_0, y_0, z_0)
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :param R: major radius of the torus
    :param r: minor radius of the torus
    :param x_0: x coordinate of the center of the torus
    :param y_0: y coordinate of the center of the torus
    :param z_0: z coordinate of the center of the torus
    :return: signed distance from the surface of the torus
    """

    # ###############
    # TODO: Implement
    
    a = np.sqrt((x-x_0)**2 + (z-z_0)**2) - R
    dist_to_surface = np.sqrt(a**2 + (y-y_0)**2) - r
    return dist_to_surface
    # ###############


def signed_distance_atom(x, y, z):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a hydrogen atom consisting of a spherical proton, a torus orbit, and one spherical electron
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :return: signed distance from the surface of the hydrogen atom
    """

    proton_center = (0, 0, 0)
    proton_radius = 0.1
    orbit_radius = 0.35
    orbit_thickness = 0.01
    electron_center = (orbit_radius, 0, 0)
    electron_radius = 0.05

    # ###############
    # TODO: Implement

    dist_to_atom = signed_distance_sphere(x, y, z, proton_radius, proton_center[0], proton_center[1], proton_center[2])
    dist_to_electron = signed_distance_sphere(x, y, z, electron_radius, electron_center[0], electron_center[1], electron_center[2])
    dist_to_orbit = signed_distance_torus(x, y, z, orbit_radius, orbit_thickness, proton_center[0], proton_center[1], proton_center[2])

    return np.min([dist_to_orbit, dist_to_electron, dist_to_atom], axis=0)
    # ###############
