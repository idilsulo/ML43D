"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    # ###############
    f = open(path, "w")
    for x, y, z in vertices:
        f.write("v {} {} {}\n".format(x, y, z))

    for v_1, v_2, v_3 in faces:     
        f.write("f {} {} {}\n".format(v_1+1, v_2+1, v_3+1))

    f.close()

    print("Exported mesh object to {}.".format(path))


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    # ###############
