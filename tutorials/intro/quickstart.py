#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from pycellfit.mesh import Mesh
from pycellfit.utils import read_segmented_image


def main():
    filename = 'hex.tif'
    fig, ax = plt.subplots()
    # load input image as a numpy ndarray
    array_of_pixels = read_segmented_image(filename, visualize=True)

    # print(img_array)

    hex_mesh = Mesh(array_of_pixels)

    hex_mesh.find_cells_from_array()

    print('there are this many cells:')

    print(hex_mesh.number_of_cells)

    hex_mesh.add_edge_points_and_junctions(array_of_pixels)

    print('number of triple junctions:')
    print(hex_mesh.number_of_triple_junctions)
    for junction in hex_mesh.junctions:
        junction.plot()

    plt.show()

    # plot the image
    fig, ax = plt.subplots()
    plt.imshow(array_of_pixels, cmap='gray', interpolation="nearest")

    for junction in hex_mesh.junctions:
        junction.plot()

    plt.xlim(215, 275)
    plt.ylim(370, 425)

    ax.set_xticks(np.arange(215, 275, 1), minor=True)
    ax.set_yticks(np.arange(370, 425, 1), minor=True)
    plt.grid(which='both', alpha=0.2)
    # plt.show()

    sum = 0
    for cell in hex_mesh.cells:
        cell.make_edges()
        # print(cell.label)
        # if cell.label == 5:
        # print('showing')
        sum += len(cell._cell_boundary_segments)
        for segment in cell._cell_boundary_segments:
            plt.plot(*zip(*segment))
    # plt.show()
    plt.savefig('res.png', dpi=1000)
    print(sum)


if __name__ == '__main__':
    main()
