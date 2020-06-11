from shapely.geometry import Point, LineString
from shapely.ops import split, linemerge

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

    def contains_triple_junction(self, linestring):
        """helper function that tells you if a shapely LineString contains a triple junction

        :param linestring: the LineString of interest
        :type linestring: Shapely LineString
        :return: if the linestring contains a junction, the first point is returned. If not,
        None is returned.
        :rtype: Shapely Point
        """
        list_of_points = []
        for junction in self.junctions:
            list_of_points.append(Point(junction.x, junction.y))

        # iterates through each point in the list
        for point in list_of_points:
            # checks if line string of interest contains the point
            if linestring.contains(point):
                return point
        return None

    def make_segments(self, cell_boundary):
        """recursive function that splits up a cell boundary based on triple junctions

        :param cell_boundary: boundary of a cell that needs to be broken up into segments
        :type cell_boundary: Shapely Linestring
        :param list_of_triple_junctions: list of all triple junctions in the mesh
        :type list_of_triple_junctions: list of Shapely Point objects
        :return results: list of segments that make up the cell boundary
        :rtype: list of Shapely LineString objects
        """
        results = []
        triple_junction = self.contains_triple_junction(cell_boundary)
        if triple_junction:
            if triple_junction == Point(list(cell_boundary.coords)[0]):
                new_boundary = LineString(list(cell_boundary.coords)[1:])
                triple_junction = self.contains_triple_junction(new_boundary)
            segments = split(cell_boundary, triple_junction).geoms
            for segment in segments:
                results += self.make_segments(segment)
        else:
            results.append(cell_boundary)
        return results

    def separate_cell_edge_points_into_segments(self):
        list_of_tjs = []
        for junction in self.junctions:
            list_of_tjs.append(Point(junction.x, junction.y))
        for cell in self.cells:
            cell.create_shapely_object()
            line = LineString(cell._shapely_object.exterior)
            answer = self.make_segments(line)
            # fixing bug: combining first and last segment
            new_answer = []
            # todo: put this in a function
            first_point_in_first_edge = Point(list(answer[0].coords)[0])
            last_point_in_last_edge = Point(list(answer[-1].coords)[-1])
            if not (first_point_in_first_edge in list_of_tjs) and not (last_point_in_last_edge in list_of_tjs):

                combined = linemerge([answer[-1], answer[0]])
                new_answer.append(combined)
                for i in range(1, len(answer) - 1):
                    new_answer.append(answer[i])
            else:
                for edge in answer:
                    new_answer.append(edge)

            # new_answer is now a list of linestrings, each line string is an edge
            cell._cell_boundary_segments = new_answer
