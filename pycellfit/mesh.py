
from . import cell, junction


class Mesh:
    def __init__(self):
        self.cells = set()
        self.edges = set()
        self.junctions = set()

    def add_cell(self, cell_pixel_value):
        self.cells.add(cell.Cell(cell_pixel_value))

    def remove_cell(self, cell_pixel_value):
        self.cells.discard(cell.Cell(cell_pixel_value))

    def add_junction(self, coordinates, degree, neighboring_cell_set):
        self.junctions.add(junction.Junction(coordinates, degree, neighboring_cell_set))

    def sorted_junctions_list(self):
        return sorted(self.junctions, key=self.sort_junction)

    def sort_junction(self, junction):
        return junction.x

    def map_junctions_to_cells(self):
        for junction in self.junctions:
            cell_labels = junction._cell_labels
            for cell in self.cells:
                if cell.label in cell_labels:
                    cell.add_junction_point((junction.x, junction.y))

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
        for junction in self.junctions:
            if junction.degree == 3:
                count += 1
        return count


