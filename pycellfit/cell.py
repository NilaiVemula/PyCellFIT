class Cell:
    def __init__(self, id_):
        self.id = id_

        # list of tuples
        self.edge_point_list = []
        self.cw_edge_point_list = []

    def add_edge_point(self, edge_point):
        self.edge_point_list.append(edge_point)

    def remove_duplicate_edge_points(self):
        self.edge_point_list = list(set(self.edge_point_list))

    def sort_edge_points_ccw(self):
        self.remove_duplicate_edge_points()

    def __str__(self):
        return 'Cell {}'.format(self.id)
