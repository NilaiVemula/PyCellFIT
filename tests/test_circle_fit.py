"""
    This file contains test cases for pycellfit/circle_fit.py file
"""

import math
import unittest

import numpy as np

from pycellfit.circle_fit import fit, distance


class TestUtils(unittest.TestCase):
    """
        This class contains test cases for the fitting process in circle_fit.py
    """

    def test1(self):
        # first quadrant
        x = np.array([0, 0.5, math.sqrt(2) / 2, math.sqrt(3) / 2, 1])
        y = np.array([1, math.sqrt(3) / 2, math.sqrt(2) / 2, 0.5, 0])
        start_point = (0, 1)
        end_point = (1, 0)

        xc, yc, radius = fit(x, y, start_point, end_point)
        self.assertAlmostEqual(xc, 0)
        self.assertAlmostEqual(yc, 0)
        self.assertAlmostEqual(radius, 1)

        center = (xc, yc)
        self.assertAlmostEqual(distance(center, start_point), distance(center, end_point))

    def test2(self):
        x = np.array([196.5, 204.5, 211.5, 219.5])
        y = np.array([370.5, 374.5, 379.5, 383.5])
        start_point = (196.5, 370.5)
        end_point = (219.5, 383.5)

        xc, yc, radius = fit(x, y, start_point, end_point)
        self.assertLess(xc, -1000000)
        self.assertGreater(yc, 1000000)
        self.assertGreater(radius, 1000000)

        center = (xc, yc)
        self.assertAlmostEqual(distance(center, start_point), distance(center, end_point))


if __name__ == "__main__":
    unittest.main()
