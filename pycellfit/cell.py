import math


class Cell:

    def __init__(self, pixel_value):
        """ constructor for a Cell object

        :param pixel_value: value of all of pixels that make up this Cell in the array
        :type pixel_value: float
        """
        # identify each cell based on its pixel value
        self._label = pixel_value

        # list of tuples of points in cell boundary
        self._edge_point_list = []

        # list of edges that make up the cell
        self._edges = []

    def add_edge_point(self, edge_point):
        if edge_point not in self._edge_point_list:
            self._edge_point_list.append(edge_point)

    @property
    def number_of_edge_points(self):
        """ returns the number of edge points in edge_point_list

        :return: number of edge points
        """

        return len(self._edge_point_list)

    @property
    def label(self):
        """ the label of a Cell is it's unique pixel value. It is assigned when the Cell object is created.

        :return:
        """
        return self._label

    @property
    def edge_points_cw(self):
        """ sort all edge points in clockwise order and return the sorted list

        :return: sorted list of all edge points (tuples)
        :rtype: list
        """

        return sorted(self._edge_point_list, key=self.clockwiseangle_and_distance)

    def approximate_cell_center(self):
        """ approximates the coordinates of the center of the cell by averaging the coordinates of points on the
        perimeter (edge) of the cell

        :return approximate center of the cell
        :rtype: tuple
        """

        xsum = 0
        ysum = 0
        for point in self._edge_point_list:
            xsum += point[0]
            ysum += point[1]
        xc = xsum / len(self._edge_point_list)
        yc = xsum / len(self._edge_point_list)
        return xc, yc

    def clockwiseangle_and_distance(self, point):
        """ helper function used in sorting edge points. Calculates the clockwise angle and the distance of a point
        from the approximate center of the cell.
        Source: https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise
        -angle-using-python

        :return: direction (clockwise angle), length vector (distance from center)
        :rtype: tuple
        """

        # origin is the approximate center of the cell
        origin = self.approximate_cell_center()

        # refvec is the order that we want to sort in
        refvec = [0, 1]  # [0,1] means clockwise

        # Vector between point and the origin: v = p - o
        vector = [point[0] - origin[0], point[1] - origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2 * math.pi + angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    def __str__(self):
        return str('Cell {}'.format(self._label))

    def __repr__(self):
        return repr('Cell {}'.format(self._label))

    def __eq__(self, other):
        return math.isclose(self.approximate_cell_center()[0], other.approximate_cell_center()[0]) and math.isclose(
            self.approximate_cell_center()[1], other.approximate_cell_center()[1])

    def __hash__(self):
        return hash(str(self))
