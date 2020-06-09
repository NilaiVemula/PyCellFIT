import itertools


class Junction:
    id_iter = itertools.count()

    def __init__(self, coordinates):
        self._coordinates = coordinates
        self._edge_labels = set()
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
        """set of labels of edges connected to this node"""

        return self._edge_labels

    def add_edge(self, edge_label):
        """Adds edge label to set of edge labels"""
        self._edge_labels.add(edge_label)

    def remove_edge(self, edge_label):
        """Remove an edge and tension vector connected to this node

        :param self:
        :param edge_label:
        """

        try:
            self._edge_labels.remove(edge_label)
        except ValueError:
            raise ValueError("{} is not connected to this Junction".format(edge_label))

    @property
    def tension_vectors(self):
        """ returns list of Tension vectors connected to this node"""

        tension_vectors = []
        for edge in self._edge_labels:
            tension_vectors.append(edge.corresponding_tension_vector)

        return tension_vectors

    @property
    def degree(self):
        return len(self._edge_labels)

    def __eq__(self, other):
        return self._coordinates == other.coordinates

    def __str__(self):
        return str(self._coordinates)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return repr('Junction({})'.format(self._coordinates))
