import itertools

from . import junction


class Edge:
    id_iter = itertools.count()

    def __init__(self, start_node, end_node, radius, center):
        self._start_node = start_node
        self._end_node = end_node
        self._radius = radius
        self._center = center
        self._intermediate_points = []
        self._mesh_points = []
        self._junctions = [start_node, end_node]
        self._cells = []
        self._label = next(Edge.id_iter)

    @property
    def start_node(self):
        return self._start_node

    @start_node.setter
    def start_node(self, node):
        if isinstance(node, junction.Junction):
            self._start_node = node
        else:
            raise TypeError('node should be of type Junction. Instead, node was of type {}'.format(type(node)))

    @property
    def end_node(self):
        return self._end_node

    @end_node.setter
    def end_node(self, node):
        if isinstance(node, junction.Junction):
            self._end_node = node
        else:
            raise TypeError('node should be of type Junction. Instead, node was of type {}'.format(type(node)))

    @property
    def radius(self):
        return self._radius[0]

    @radius.setter
    def radius(self, r):
        if isinstance(r, (int, float, complex)) and not isinstance(r, bool):
            self._radius = r
        else:
            raise TypeError('radius must be of numeric type. Instead, r was of type {}'.format(type(r)))

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, c):
        if len(c) == 2:
            self._center = c
        else:
            raise ValueError('center should not exceed length of 2. The length of center coordinates was: {}'.format(
                len(c)))

    @property
    def xc(self):
        return self._center[0]

    @property
    def yc(self):
        return self._center[1]

    def __eq__(self, other):
        return self._start_node == other.start_node and self._end_node == other.end_node and self._center == \
               other.center

    def __str__(self):
        return str(self._start_node) + ' to ' + str(self._end_node)

    def __hash__(self):
        return hash(str(self))
