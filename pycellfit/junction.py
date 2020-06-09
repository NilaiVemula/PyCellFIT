import itertools


class Junction:
    id_iter = itertools.count()

    def __init__(self, coordinates):
        self._coordinates = coordinates
        self._edges = []
        self._label = next(Junction.id_iter)

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        if len(coordinates) == 2:
            self._coordinates = coordinates
        else:
            raise ValueError('coordinates should not exceed length of 2. The length of coordinates was: {}'.format(
                len(coordinates)))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

    @property
    def edges(self):
        """List of edges connected to this node"""

        return self._edges

    def add_edge(self, edge):
        """Adds edge to list of edges -- make sure no repeat edges"""
        if edge not in self._edges:
            self._edges.append(edge)

    def remove_edge(self, edge):
        """Remove an edge and tension vector connected to this node
        :param edge:
        """

        try:
            self._edges.remove(edge)
        except ValueError:
            raise ValueError("{} is not connected to this Junction".format(edge))

    @property
    def tension_vectors(self):
        """ returns list of Tension vectors connected to this node"""

        tension_vectors = []
        for edge in self._edges:
            tension_vectors.append(edge.corresponding_tension_vector)

        return tension_vectors

    @property
    def degree(self):
        return len(self._edges)

    def __eq__(self, other):
        return self._coordinates == other.coordinates

    def __str__(self):
        return str(self._coordinates)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return repr('Junction({})'.format(self._coordinates))
