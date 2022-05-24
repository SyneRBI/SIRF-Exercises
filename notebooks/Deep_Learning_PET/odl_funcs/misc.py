# This is functionality extracted from [Operator Discretization Library (ODL)]
# (https://odlgroup.github.io/odl/index.html) and changed somewhat for our needs.
# The appropriate pieces of code that are used are: [here]
# (https://github.com/odlgroup/odl/blob/master/odl/phantom/transmission.py)
# and [here](https://github.com/odlgroup/odl/blob/master/odl/phantom/geometric.py).

# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


def _getshapes_2d(center, max_radius, shape):
    """Calculate indices and slices for the bounding box of a disk."""
    index_mean = shape * center
    index_radius = max_radius / 2.0 * np.array(shape)

    # Avoid negative indices
    min_idx = np.maximum(np.floor(index_mean - index_radius), 0).astype(int)
    max_idx = np.ceil(index_mean + index_radius).astype(int)
    idx = [slice(minx, maxx) for minx, maxx in zip(min_idx, max_idx)]
    shapes = [(idx[0], slice(None)),
              (slice(None), idx[1])]
    return tuple(idx), tuple(shapes)

def ellipse_phantom(shape, ellipses):
    
    """Create a phantom of ellipses in 2d space.

    Parameters
    ----------
    shape : `tuple`
        Size of image
    ellipses : list of lists
        Each row should contain the entries ::

            'value',
            'axis_1', 'axis_2',
            'center_x', 'center_y',
            'rotation'

        The provided ellipses need to be specified relative to the
        reference rectangle ``[-1, -1] x [1, 1]``. Angles are to be given
        in radians.

    Returns
    -------
    phantom : numpy.ndarray
        2D ellipse phantom.

    See Also
    --------
    shepp_logan : The typical use-case for this function.
    """
    # Blank image
    p = np.zeros(shape)

    grid_in = (np.expand_dims(np.linspace(0, 1, shape[0]),1),
                    np.expand_dims(np.linspace(0, 1, shape[1]),0))

    # move points to [-1, 1]
    grid = []
    for i in range(2):
        mean_i = 0.5
        # Where space.shape = 1, we have minp = maxp, so we set diff_i = 1
        # to avoid division by zero. Effectively, this allows constructing
        # a slice of a 2D phantom.
        diff_i = 0.5
        grid.append((grid_in[i] - mean_i) / diff_i)

    for ellip in ellipses:
        assert len(ellip) == 6

        intensity = ellip[0]
        a_squared = ellip[1] ** 2
        b_squared = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        theta = ellip[5]

        scales = [1 / a_squared, 1 / b_squared]
        center = (np.array([x0, y0]) + 1.0) / 2.0

        # Create the offset x,y and z values for the grid
        if theta != 0:
            # Rotate the points to the expected coordinate system.
            ctheta = np.cos(theta)
            stheta = np.sin(theta)

            mat = np.array([[ctheta, stheta],
                            [-stheta, ctheta]])

            # Calculate the points that could possibly be inside the volume
            # Since the points are rotated, we cannot do anything directional
            # without more logic
            max_radius = np.sqrt(
                np.abs(mat).dot([a_squared, b_squared]))
            idx, shapes = _getshapes_2d(center, max_radius, shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            offset_points = [vec * (xi - x0i)[..., None]
                             for xi, vec, x0i in zip(subgrid,
                                                     mat.T,
                                                     [x0, y0])]
            rotated = offset_points[0] + offset_points[1]
            np.square(rotated, out=rotated)
            radius = np.dot(rotated, scales)
        else:
            # Calculate the points that could possibly be inside the volume
            max_radius = np.sqrt([a_squared, b_squared])
            idx, shapes = _getshapes_2d(center, max_radius, shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            squared_dist = [ai * (xi - x0i) ** 2
                            for xi, ai, x0i in zip(subgrid,
                                                   scales,
                                                   [x0, y0])]

            # Parentheses to get best order for broadcasting
            radius = squared_dist[0] + squared_dist[1]

        # Find the points within the ellipse
        inside = radius <= 1

        # Add the ellipse intensity to those points
        p[idx][inside] += intensity
    return p

def random_shapes():
    x_0 = 1 * np.random.rand() - 0.5
    y_0 = 1 * np.random.rand() - 0.5
    return [np.random.exponential(0.4),
            1 * np.random.rand() - 0.5, 1 * np.random.rand() - 0.5,
            x_0, y_0,
            np.random.rand() * 2 * np.pi]

def random_phantom(space, n_ellipse=20):
    n = np.random.poisson(n_ellipse)
    shapes = [random_shapes() for _ in range(n)]
    for i in range(n):
        shapes[i][0] = np.random.exponential(0.4)
    x = ellipse_phantom(space[1:], shapes)
    x = [x]
    return np.array(x)

def shepp_logan(space):
    rad18 = np.deg2rad(18.0)
    #            value  axisx  axisy     x       y  rotation
    ellipsoids= [[0.55, 0.69, 0.92, 0.0, 0.0, 0],
                [0.60, 0.6624, 0.874, 0.0, -0.0184, 0],
                [0.50, 0.11, 0.31, 0.22, 0.0, -rad18],
                [0.51, 0.16, 0.41, -0.22, 0.0, rad18],
                [0.05, 0.21, 0.25, 0.0, 0.35, 0],
                [0.11, 0.046, 0.046, 0.0, 0.1, 0],
                [0.48, 0.046, 0.046, 0.0, -0.1, 0],
                [0.34, 0.046, 0.023, -0.08, -0.605, 0],
                [0.14, 0.023, 0.023, 0.0, -0.606, 0],
                [1.28, 0.023, 0.046, 0.06, -0.605, 0]]
    x = ellipse_phantom(space[1:], ellipsoids)
    x = [x]
    return np.array(x)