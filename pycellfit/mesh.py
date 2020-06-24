import math
import statistics

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .cell import Cell
from .junction import Junction


class Mesh:
    def __init__(self, array_of_pixels):
        self.cells = set()
        self.edges = set()
        self.junctions = set()
        self.array_of_pixels = array_of_pixels
        self.points = set()
        self.background_label = 0

    def add_cell(self, cell_pixel_value):
        self.cells.add(Cell(cell_pixel_value))

    def remove_cell(self, cell_pixel_value):
        self.cells.discard(Cell(cell_pixel_value))

    def find_cells_from_array(self):
        # find all unique pixel values in array
        cell_ids = set()
        for row, col in np.ndindex(self.array_of_pixels.shape):
            cell_ids.add(self.array_of_pixels[row, col])

        # determine pixel value of background and remove it from our set of cell ids
        # potential background values are the values in the four corners of the array
        potential_background_values = self.array_of_pixels[[0, 0, -1, -1], [0, -1, 0, -1]]
        # we determine the background value to be the mode of the potential values
        background_value = stats.mode(potential_background_values)[0][0]
        cell_ids.remove(background_value)
        self.background_label = background_value
        for cell_id in cell_ids:
            self.add_cell(cell_id)

    @property
    def number_of_cells(self):
        """ returns the number of cells in the mesh

        :return: number of cells in mesh
        :rtype: int
        """

        return len(self.cells)

    @property
    def number_of_edges(self):
        """ returns the number of edges in the mesh

        :return: number of edges in the mesh
        :rtype: int
        """

        return len(self.edges)

    @property
    def number_of_junctions(self):
        """ returns the number of junctions in the mesh

        :return: number of junctions in the mesh
        :rtype: int
        """

        return len(self.junctions)

    @property
    def number_of_triple_junctions(self):
        """ counts and outputs the number of triple junctions in the mesh

        :return number of triple junctions in mesh
        :rtype: int
        """

        count = 0
        for j in self.junctions:
            if j.degree == 3:
                count += 1
        return count

    def add_edge_points_and_junctions(self, array_of_pixels):
        with np.nditer(array_of_pixels, flags=['multi_index']) as iterator:
            for pixel in iterator:

                # find location of this pixel and the surrounding pixels
                position = iterator.multi_index
                north = tuple(map(lambda i, j: i + j, position, (-1, 0)))
                west = tuple(map(lambda i, j: i + j, position, (0, -1)))
                south = tuple(map(lambda i, j: i + j, position, (1, 0)))
                east = tuple(map(lambda i, j: i + j, position, (0, 1)))
                southeast = tuple(map(lambda i, j: i + j, position, (1, 1)))

                # find triple junctions using 2*2 region of array
                try:
                    neighboring_values = {array_of_pixels[east],
                                          array_of_pixels[south],
                                          array_of_pixels[southeast],
                                          array_of_pixels[position]}
                except IndexError:
                    pass
                if len(neighboring_values) == 3:
                    x = position[1] + 1 - 0.5
                    y = position[0] + 1 - 0.5
                    j = Junction((x, y), neighboring_values)
                    self.junctions.add(j)
                    for cell in self.cells:
                        if cell.label in neighboring_values:
                            cell.junctions.add(j)

                # find edge points using four neighbors
                try:
                    if array_of_pixels[position] != array_of_pixels[east]:
                        for cell in self.cells:
                            if cell.label == pixel:
                                cell.add_edge_point((position[1] + 1 - 0.5, position[0] - 0.5), array_of_pixels[east])
                                cell.add_edge_point((position[1] + 1 - 0.5, position[0] + 1 - 0.5),
                                                    array_of_pixels[east])
                except IndexError:
                    pass
                try:
                    if array_of_pixels[position] != array_of_pixels[south]:
                        for cell in self.cells:
                            if cell.label == pixel:
                                cell.add_edge_point((position[1] + 1 - 0.5, position[0] + 1 - 0.5),
                                                    array_of_pixels[south])
                                cell.add_edge_point((position[1] - 0.5, position[0] + 1 - 0.5), array_of_pixels[south])
                except IndexError:
                    pass
                try:
                    if array_of_pixels[position] != array_of_pixels[north]:
                        for cell in self.cells:
                            if cell.label == pixel:
                                cell.add_edge_point((position[1] + 1 - 0.5, position[0] - 0.5), array_of_pixels[north])
                                cell.add_edge_point((position[1] + 1 - 0.5, position[0] - 0.5), array_of_pixels[north])
                except IndexError:
                    pass
                try:
                    if array_of_pixels[position] != array_of_pixels[west]:
                        for cell in self.cells:
                            if cell.label == pixel:
                                cell.add_edge_point((position[1] - 0.5, position[0] - 0.5), array_of_pixels[west])
                                cell.add_edge_point((position[1] - 0.5, position[0] + 1 - 0.5), array_of_pixels[west])
                except IndexError:
                    pass

    def make_edges_for_all_cells(self):
        for cell in self.cells:
            cell.make_edges(self.edges)

    def circle_fit_all_edges(self):
        for edge in self.edges:
            edge.circle_fit()

    def generate_mesh(self, average_nodes_per_edge=4):
        edge_lengths = []
        for edge in self.edges:
            edge_lengths.append(edge.length)
        distance_between_nodes = [edge_length / (average_nodes_per_edge - 1) for edge_length in edge_lengths]
        # mean_edge_length = sum(edge_lengths)/len(edge_lengths)
        # mean_edge_length = statistics.median(edge_lengths)
        # length = mean_edge_length/(average_nodes_per_edge-1)
        length = statistics.mean(distance_between_nodes)
        print(length)
        for edge in self.edges:
            edge.split_line_multiple(n_pieces=math.ceil(edge.length / length))
            edge.calculate_edge_points()
            for point in edge._mesh_points:
                self.points.add(point)

    def plot(self):
        for point in self.points:
            plt.scatter(point[0], point[1], c='green', s=1)

    def solve_tensions(self):
        edge_label_to_tension_label_dict = {}  # each entry is edge label: tension_label
        n_tensions = 0
        for edge in self.edges:
            if not edge.outside(self.background_label):
                edge.tension_label = n_tensions
                edge_label_to_tension_label_dict[edge._label] = n_tensions
                edge.map_unit_vectors_to_junctions()
                n_tensions += 1
        gy_matrix = np.zeros((2 * self.number_of_triple_junctions, n_tensions))
        for junction in self.junctions:
            for edge_label in junction.x_unit_vectors_dict:
                tension_label = edge_label_to_tension_label_dict[edge_label]
                gy_matrix[junction._label][tension_label] = junction.x_unit_vectors_dict[edge_label]
            for edge_label in junction.y_unit_vectors_dict:
                tension_label = edge_label_to_tension_label_dict[edge_label]
                gy_matrix[junction._label + self.number_of_junctions][tension_label] = junction.y_unit_vectors_dict[
                    edge_label]
        print(gy_matrix.shape)
        top_left = np.matmul(gy_matrix.transpose(), gy_matrix)
        bottom_left = np.full((1, np.shape(top_left)[1]), 1)
        top_right = bottom_left.transpose()
        top = np.concatenate((top_left, top_right), axis=1)
        bottom_right = np.zeros((1, top.shape[1] - bottom_left.shape[1]))
        bottom = np.concatenate((bottom_left, bottom_right), axis=1)
        big_matrix = np.concatenate((top, bottom), axis=0)
        zero = np.zeros((n_tensions + 1, 1))
        zero[261][0] = 261
        print(big_matrix.shape, zero.shape)
        y = np.linalg.lstsq(big_matrix, zero, rcond=None)
        # print(y[0])
        print(np.mean(y[0]))
        for edge_label, tension_label in edge_label_to_tension_label_dict.items():
            tension_magnitude = y[0][tension_label]
            for edge in self.edges:
                if edge._label == edge_label:
                    edge.tension_magnitude = tension_magnitude

    def plot_tensions(self):
        max_tension = 2
        from matplotlib import cm

        viridis = cm.get_cmap('Spectral', 12)
        for edge in self.edges:
            if not edge.outside(self.background_label):
                edge.plot_tangent(c=viridis(edge.tension_magnitude / max_tension)[0])
