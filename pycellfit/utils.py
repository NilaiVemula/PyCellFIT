""" Top-level utility functions for pycellfit module."""

__author__ = "Nilai Vemula"

import os

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import geometry
from shapely import ops


def read_segmented_image(file_name, visualize=False):
    """Displays the segmented image using matplotlib

    :param file_name: file name of a segmented image in .tif format
    :type file_name: str
    :param visualize: if true, then image will be plotted using matplotlib.pyplot
    :type visualize: bool
    :raises TypeError: only accepts tif/tiff files as input
    :return: array of pixel values
    :rtype: numpy.ndarray
    """

    extension = os.path.splitext(file_name)[1]
    if not (extension == '.tif' or extension == '.tiff'):
        raise TypeError("Invalid File Type. File must be a .tif or .tiff")

    im = PIL.Image.open(file_name)
    img_array = np.array(im)

    if visualize:
        plt.imshow(img_array, cmap='gray', interpolation="nearest", vmax=255)

    return img_array


def contains_triple_junction(linestring, list_of_points):
    """helper function that tells you if a shapely LineString contains a triple junction

    :param linestring: the LineString of interest
    :type linestring: shapely.geometry.LineString
    :param list_of_points: list containing a bunch of points of type :class:`shapely.geometry.Point`
    :type list_of_points: list
    :return: if the linestring contains a point in the list_of_points, the first point is returned; else, None.
    :rtype: shapely.geometry.Point
    """

    # iterates through each point in the list
    for point in list_of_points:
        # checks if line string of interest contains the point
        if linestring.contains(point):
            return point
    return None


def make_segments(cell_boundary, list_of_triple_junctions):
    """recursive function that splits up a cell boundary based on triple junctions

    :param cell_boundary: boundary of a cell that needs to be broken up into segments
    :type cell_boundary: shapely.geometry.LineString
    :param list_of_triple_junctions: list of all triple junctions (of type :class:`shapely.geometry.Point`) in the mesh
    :type list_of_triple_junctions: list
    :return results: list of segments (of type :class:`shapely.geometry.LineString`) that make up the cell boundary
    :rtype: list
    """

    results = []
    triple_junction = contains_triple_junction(cell_boundary, list_of_triple_junctions)
    if triple_junction:
        if triple_junction == geometry.Point(list(cell_boundary.coords)[0]):
            new_boundary = geometry.LineString(list(cell_boundary.coords)[1:])
            triple_junction = contains_triple_junction(new_boundary, list_of_triple_junctions)
        segments = ops.split(cell_boundary, triple_junction).geoms
        for segment in segments:
            results += make_segments(segment, list_of_triple_junctions)
    else:
        results.append(cell_boundary)
    return results


def extract_all_points(dataframe_of_cells, visualize=False):
    """generate list of all points in mesh

    :param visualize: boolean to indicate if plot of all points should be made or not
    :type visualize: bool
    :param dataframe_of_cells: a dataframe where each row contains a unique cell
    :type dataframe_of_cells: geopandas dataframe
    :return: list_of_all_points: list of all points in the mesh
    :rtype: list
    """

    # make empty np arrays
    all_points_x = np.empty(1)
    all_points_y = np.empty(1)
    # fill np arrays with all x and y points in the boundaries of all cells in the mesh
    for index, row in dataframe_of_cells.iterrows():
        x, y = row['geometry'].boundary.xy
        x = np.asarray(x)
        y = np.asarray(y)

        all_points_x = np.concatenate((all_points_x, x), axis=0)
        all_points_y = np.concatenate((all_points_y, y), axis=0)

    # combine x and y points to one np array
    all_points = np.column_stack((all_points_x, all_points_y))

    # removed duplicate points
    all_points = np.unique(all_points, axis=0)
    all_x_points = all_points[:, 0].tolist()
    all_y_points = all_points[:, 1].tolist()
    list_of_all_points = []
    for (x, y) in zip(all_x_points, all_y_points):
        list_of_all_points.append(geometry.Point(x, y))

    if visualize:
        # Visualize all points
        xs = [point.x for point in list_of_all_points]
        ys = [point.y for point in list_of_all_points]
        plt.scatter(xs, ys)
        plt.show()

    return list_of_all_points


def locate_triple_junctions(dataframe_of_cells, visualize=False):
    """generate list of triple junctions in mesh

    :param visualize: boolean to indicate if plot of all points should be made or not
    :type visualize: bool
    :param dataframe_of_cells: a dataframe where each row contains a unique cell
    :type dataframe_of_cells: geopandas dataframe
    :return: list_of_tjs: list of all triple junctions in the mesh
    :rtype: list
    """

    list_of_all_points = extract_all_points(dataframe_of_cells, visualize)

    nparray = np.empty((len(list_of_all_points), len(dataframe_of_cells.geometry)))
    for j, cell in enumerate(dataframe_of_cells.geometry):
        line = geometry.LineString(cell.exterior)
        for i, point in enumerate(list_of_all_points):
            nparray[i, j] = (line.contains(point))

    num_of_junctions = nparray.sum(axis=1).tolist()

    df = pd.DataFrame(list(zip(list_of_all_points, num_of_junctions)), columns=['Points', 'Number of Junctions'])
    df = df.assign(is_tj=(df['Number of Junctions'] == 3))
    tjs = df.query('is_tj == True')
    list_of_tjs = tjs.Points.tolist()

    if visualize:
        # Visualize all points
        xs = [point.x for point in list_of_tjs]
        ys = [point.y for point in list_of_tjs]
        plt.scatter(xs, ys)
        plt.show()

    return list_of_tjs
