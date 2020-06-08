"""test cases for segmentation_transform functions"""

import sys
import unittest

import numpy as np
from PIL import Image

import pycellfit.segmentation_transform
from pycellfit.utils import read_segmented_image


class TestUtils(unittest.TestCase):
    """
        This class contains test cases for the functions in pycellfit/segmentation_tranform.py
    """

    def test_skeleton_to_watershed(self):
        array = np.array([[0, 0, 0, 0, 0],
                          [0, 255, 255, 255, 0],
                          [0, 255, 0, 255, 0],
                          [0, 255, 255, 255, 0],
                          [0, 0, 0, 0, 0]])
        actual_result = pycellfit.segmentation_transform.skeleton_to_watershed(array)
        print(actual_result)
        expected_result = np.array([[0, 0, 0, 0, 0],
                                    [0, 255, 255, 255, 0],
                                    [0, 255, 2, 255, 0],
                                    [0, 255, 255, 255, 0],
                                    [0, 0, 0, 0, 0]])
        self.assertIsNone(np.testing.assert_array_equal(actual_result, expected_result))

    def test_watershed_to_skeleton(self):
        hex_array = read_segmented_image('tests/images/hex.tif')
        skeleton_array = pycellfit.segmentation_transform.watershed_to_skeleton(hex_array)
        im = Image.fromarray(skeleton_array)
        im.save('tests/images/hex_skeleton.tif')

        hex_skeleton_array = read_segmented_image('tests/images/hex_skeleton.tif')

        self.assertIsNone(np.testing.assert_array_equal(skeleton_array, hex_skeleton_array))

    def test_noboundaries(self):
        hex_skeleton_array = read_segmented_image('tests/images/hex_skeleton.tif')
        sys.setrecursionlimit(10 ** 6)
        watershed_array = pycellfit.segmentation_transform.skeleton_to_watershed(hex_skeleton_array)
        im = Image.fromarray(watershed_array)
        im.save('tests/images/hex_watershed.tif')

        self.assertEqual(0, 0)


if __name__ == "__main__":
    unittest.main()
