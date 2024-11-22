# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from gradbench.adbench.ht_data import HandData, HandInput, HandModel

DELIM = ":"


def parse_floats(arr):
    """Parses enumerable as float.

    Args:
        arr (enumerable): input data that can be parsed to floats.

    Returns:
        (List[float]): parsed data.
    """

    return [float(x) for x in arr]


def load_model(path):
    """Loads HandModel from the given file.

    Args:
        path(str): path to a directory with input files.

    Returns:
        (HandModel): hand trcking model.
    """

    # Read in triangle info.
    triangles = np.loadtxt(os.path.join(path, "triangles.txt"), int, delimiter=DELIM)

    # Process bones file.
    bones_path = os.path.join(path, "bones.txt")

    # Grab bone names.
    bone_names = tuple(line.split(DELIM)[0] for line in open(bones_path))

    # Grab bone parent indices.
    parents = np.loadtxt(bones_path, int, usecols=[1], delimiter=DELIM).flatten()

    # Grab relative transforms.
    relative_transforms = np.loadtxt(
        bones_path, usecols=range(2, 2 + 16), delimiter=DELIM
    ).reshape(len(parents), 4, 4)

    vertices_path = os.path.join(path, "vertices.txt")
    n_bones = len(bone_names)

    # Find number of vertices.
    with open(vertices_path) as handle:
        n_verts = len(handle.readlines())

    # Read in vertex info.
    positions = np.zeros((n_verts, 3))
    weights = np.zeros((n_verts, n_bones))

    with open(vertices_path) as handle:
        for i_vert, line in enumerate(handle):
            atoms = line.split(DELIM)
            positions[i_vert] = parse_floats(atoms[:3])

            for i in range(int(atoms[8])):
                i_bone = int(atoms[9 + i * 2])
                weights[i_vert, i_bone] = float(atoms[9 + i * 2 + 1])

    # Grab absolute invers transforms.
    inverse_absolute_transforms = np.loadtxt(
        bones_path, usecols=range(2 + 16, 2 + 16 + 16), delimiter=DELIM
    ).reshape(len(parents), 4, 4)

    n_vertices = positions.shape[0]
    homogeneous_base_positions = np.ones((n_vertices, 4))
    homogeneous_base_positions[:, :3] = positions

    result = HandModel(
        n_bones,
        bone_names,
        parents,
        relative_transforms,
        inverse_absolute_transforms,
        homogeneous_base_positions,
        weights,
        triangles,
        False,  # WARNING: not exactly understand where such info comes from
    )

    return result


def read_hand_instance(model_dir, fn, read_us):
    """Reads input data for hand tracking objective.

    Args:
        model_dir (str): path to the directory contatins model data files.
        fn (str): name of the file contains additional data for objective.
        read_us (bool): if True then complicated scheme is used.

    Returns:
        (HandInput): input data for hand objective test class.
    """

    model = load_model(model_dir)

    fid = open(fn, "r")
    line = fid.readline()
    line = line.split()
    npts = int(line[0])
    ntheta = int(line[1])

    lines = [fid.readline().split() for _ in range(npts)]
    correspondences = np.array([int(line[0]) for line in lines])
    points = np.array([parse_floats(line[1:]) for line in lines])

    if read_us:
        us = np.array([parse_floats(fid.readline().split()) for _ in range(npts)])

    params = np.array([float(fid.readline()) for _ in range(ntheta)])
    fid.close()

    data = HandData(model, correspondences, points)

    if read_us:
        return HandInput(params, data, us)
    else:
        return HandInput(params, data)
