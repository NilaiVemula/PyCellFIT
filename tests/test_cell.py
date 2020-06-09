"""
    This file contains test cases for pycellfit/cell.py file
"""

import unittest

from pycellfit.cell import Cell


class TestUtils(unittest.TestCase):
    """
        This class contains test cases for the functions describing the cell class
    """

    def test_constructor_and_label(self):
        # Case 1: first cell should have label of 0
        cell_0 = Cell(0)
        self.assertEqual(cell_0.label, 0)

        # Case 2: subsequent cell labels should auto-increment
        cell_1 = Cell(1)
        self.assertEqual(cell_1.label, 1)

    def test_add_edge_points(self):
        # Case 1: adds edge points and number of edge points increases
        cell_0 = Cell(0)
        cell_membrane = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for point in cell_membrane:
            cell_0.add_edge_point(point)
        self.assertEqual(cell_0.number_of_edge_points, 4)

        # Case 2: doesn't add duplicate points
        cell_1 = Cell(1)
        cell_membrane = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]
        for point in cell_membrane:
            cell_1.add_edge_point(point)
        self.assertEqual(cell_1.number_of_edge_points, 4)

    def test_edge_points_cw(self):
        # Case 1: empty cell should return empty list
        cell_0 = Cell(0)
        self.assertEqual(cell_0.edge_points_cw, [])

        # Case 2: sorting should work properly
        cell_1 = Cell(1)
        cell_membrane = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]
        for point in cell_membrane:
            cell_1.add_edge_point(point)
        print(type(cell_1.edge_points_cw))
        self.assertEqual(cell_1.edge_points_cw, [(1, 1), (1, 0), (0, 0), (0, 1)])

    def test_approximate_center(self):
        # Center of square
        cell_0 = Cell(0)
        cell_membrane = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for point in cell_membrane:
            cell_0.add_edge_point(point)
        self.assertEqual(cell_0.approximate_cell_center(), (0.5, 0.5))

    def test_eq(self):
        cell_0 = Cell(0)
        cell_1 = Cell(0)
        s = {cell_0, cell_1}
        self.assertEqual(len(s), 1)

    def test_add_remove_cells(self):
        s = set()
        s.add(Cell(0))
        self.assertEqual(len(s), 1)
        s.remove(Cell(0))
        self.assertEqual(len(s), 0)


if __name__ == "__main__":
    unittest.main()
