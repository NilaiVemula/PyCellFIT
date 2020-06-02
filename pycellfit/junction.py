class Junction:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self._edges = []
        self._tension_vectors = []

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
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

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

        ind = self._edges.index(edge)
        self._edges.pop(ind)
        self._tension_vectors.pop(ind)

    @property
    def tension_vectors(self):
        """ Tension vectors connected to this node"""
        return self._tension_vectors

    def add_tension_vectors(self, vector):
        """ add tension to tension_vectors, make sure no repeat tension_vectors """
        if vector not in self._tension_vectors:
            self._tension_vectors.append(vector)

    def __eq__(self, other):
        return self.coordinates == other.coordinates

    def __str__(self):
        return str(self.coordinates)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return repr('Junction({})'.format(self.coordinates))
