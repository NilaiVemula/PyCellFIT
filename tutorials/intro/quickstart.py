#!/usr/bin/env python3

import matplotlib.pyplot as plt

from pycellfit.mesh import Mesh
from pycellfit.utils import read_segmented_image


def main():
    # STEP 1: Reading in segmented image
    filename = 'hex.tif'
    # filename = 'Segment_0_000.tif'
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

    hex_mesh.generate_mesh()

    # STEP 3: Circle Fit
    hex_mesh.circle_fit_all_edges()

    # STEP ??: Visualize
    # show segmented image
    plt.imshow(array_of_pixels, cmap='gray', interpolation="nearest", vmax=255)

    # show junctions
    # for junction in hex_mesh.junctions:
    #     junction.plot(label=True)
    #
    # # show edges
    # for edge in hex_mesh.edges:
    #     if not edge.outside:
    #         edge.plot(label=True)

    # show cells
    for cell in hex_mesh.cells:
        cell.plot()
    hex_mesh.plot()
    for edge in hex_mesh.edges:
        if not edge.outside(hex_mesh.background_label):
            print(edge.start_tangent_angle, edge.end_tangent_angle)
            edge.plot_circle()
            #edge.plot_tangent()
    plt.savefig('fig.png', dpi=1000)
    # plt.show()


if __name__ == '__main__':
    main()
