#!/usr/bin/env python3

import matplotlib.pyplot as plt

from pycellfit.mesh import Mesh
from pycellfit.utils import read_segmented_image


def main():
    # STEP 1: Reading in segmented image
    filename = 'hex.tif'
    fig, ax = plt.subplots()
    # load input image as a numpy ndarray
    array_of_pixels = read_segmented_image(filename)

    # STEP 2: Generate Mesh
    hex_mesh = Mesh(array_of_pixels)

    hex_mesh.find_cells_from_array()

    print('there are this many cells:')
    print(hex_mesh.number_of_cells)

    hex_mesh.add_edge_points_and_junctions(array_of_pixels)

    print('number of triple junctions:')
    print(hex_mesh.number_of_triple_junctions)

    hex_mesh.make_edges_for_all_cells()
    print('number of edges')
    print(hex_mesh.number_of_edges)

    for edge in hex_mesh.edges:
        print(edge.length, edge.location)

    # STEP 3: Circle Fit
    # hex_mesh.circle_fit_all_edges()
    # for edge in hex_mesh.edges:
    #    print(edge.radius)


if __name__ == '__main__':
    main()
