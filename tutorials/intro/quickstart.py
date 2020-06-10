#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pycellfit.mesh import Mesh
from pycellfit.utils import read_segmented_image


def main():
    filename = 'hex.tif'
    fig, ax = plt.subplots()
    # load input image as a numpy ndarray
    array_of_pixels = read_segmented_image(filename, visualize=True)

    # print(img_array)

    # find all unique pixel values in array
    cell_ids = set()
    for row, col in np.ndindex(array_of_pixels.shape):
        cell_ids.add(array_of_pixels[row, col])

    # determine pixel value of background and remove it from our set of cell ids
    # potential background values are the values in the four corners of the array
    potential_background_values = array_of_pixels[[0, 0, -1, -1], [0, -1, 0, -1]]
    # we determine the background value to be the mode of the potential values
    background_value = stats.mode(potential_background_values)[0][0]
    cell_ids.remove(background_value)

    # create a mesh for this image
    hex_mesh = Mesh()
    for cell_id in cell_ids:
        hex_mesh.add_cell(cell_id)

    print('there are this many cells:')
    print(hex_mesh.number_of_cells)

    with np.nditer(array_of_pixels, flags=['multi_index']) as iterator:
        for pixel in iterator:

            # find pixels that are on the boundaries of a cell
            # find location of this pixel and the surrounding pixels
            position = iterator.multi_index
            north = tuple(map(lambda i, j: i + j, position, (-1, 0)))
            south = tuple(map(lambda i, j: i + j, position, (1, 0)))
            east = tuple(map(lambda i, j: i + j, position, (0, 1)))
            west = tuple(map(lambda i, j: i + j, position, (0, -1)))

            # find values of these five pixels
            try:

                north_value, east_value, west_value, south_value = (array_of_pixels[north],
                                                                    array_of_pixels[east], array_of_pixels[west],
                                                                    array_of_pixels[south])
                neighboring_values = {array_of_pixels[north], array_of_pixels[east], array_of_pixels[west],
                                      array_of_pixels[south]}
            except IndexError:
                pass
            y, x = position
            y += 0.5
            x += 0.5
            if north_value != pixel:
                y -= 0.5
            if south_value != pixel:
                y += 0.5
            if east_value != pixel:
                x += 0.5
            if west_value != pixel:
                x -= 0.5
            # neighboring_values.discard(background_value)

            if len(neighboring_values) >= 2:
                for cell in hex_mesh.cells:
                    if cell.label == pixel:
                        cell.add_edge_point((x, y))

            # find pixels that are triple junctions
            southeast = tuple(map(lambda i, j: i + j, position, (1, 1)))

            try:
                neighboring_values = {array_of_pixels[position], array_of_pixels[south], array_of_pixels[southeast],
                                      array_of_pixels[east]}
            except IndexError:
                pass
            y, x = position
            y += 1
            x += 1
            if len(neighboring_values) == 3:
                hex_mesh.add_junction((x, y), 3)

    print('number of triple junctions:')
    print(hex_mesh.number_of_triple_junctions)
    for junction in hex_mesh.junctions:
        junction.plot()
    # plt.show()
    sorted_list = hex_mesh.sorted_junctions_list()
    print(sorted_list)

    # plt.show()
    for cell in hex_mesh.cells:
        shapely = cell.create_shapely_object()
        # plt.plot(*zip(*sorted_points))
        # plt.show()
        x, y = shapely.exterior.xy
        plt.plot(x, y, c='b')
    plt.xlim(215, 270)
    plt.ylim(365, 430)

    ax.set_xticks(np.arange(215, 270, 1), minor=True)
    ax.set_yticks(np.arange(365, 430, 1), minor=True)
    plt.grid(which='both', alpha=0.2)
    plt.show()

    fig, ax = plt.subplots()
    plt.imshow(array_of_pixels, cmap='gray', interpolation="nearest")
    for junction in hex_mesh.junctions:
        junction.plot()
    hex_mesh.separate_cell_edge_points_into_segments()
    for cell in hex_mesh.cells:
        print(cell.label, len(cell._cell_boundary_segments))
        for edge in cell._cell_boundary_segments:
            x, y = edge.xy
            plt.plot(x, y)
    plt.xlim(215, 270)
    plt.ylim(365, 430)

    ax.set_xticks(np.arange(215, 270, 1), minor=True)
    ax.set_yticks(np.arange(365, 430, 1), minor=True)
    plt.grid(which='both', alpha=0.2)
    plt.show()


if __name__ == '__main__':
    main()
