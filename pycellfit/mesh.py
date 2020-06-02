class Mesh:
    def __init__(self):
        self._cells = []
        self._edges = []
        self._junctions = []

    def add_cell(self, cell):
        if cell not in self._cells:
            self._cells.append(cell)

    def remove_cell(self, cell):
        if cell in self._cells:
            self._cells.remove(cell)
