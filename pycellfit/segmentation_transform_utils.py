import numpy as np


def fill_recursively(array_of_pixels, position, old_value, new_value):
    """recursive function to replace the value of a pixel and all surrounding pixels to a new value

    :type array_of_pixels: np.ndarray
    :param array_of_pixels: 2D numpy array of all pixel values
    :param position: tuple with (row, col) location of pixel to modify
    :type position: tuple
    :type old_value: float
    :param old_value: old value of pixel at `position`
    :type new_value: float
    :param new_value: new value for pixel at `position` and surrounding pixels
    :return: None
    """

    # Base Case
    row, col = position
    if row < 0 or row >= np.shape(array_of_pixels)[0] or col < 0 or col >= np.shape(array_of_pixels)[1]:
        # out of bounds on the array
        return
    elif array_of_pixels[row][col] != old_value or array_of_pixels[row][col] == new_value:
        # pixel is in different region or has already been modified
        # pixel value does not need to be modified
        return
    else:
        # change pixel value
        array_of_pixels[row][col] = new_value

        # recurs for north, east, south, west
        fill_recursively(array_of_pixels, (row + 1, col), old_value, new_value)
        fill_recursively(array_of_pixels, (row - 1, col), old_value, new_value)
        fill_recursively(array_of_pixels, (row, col + 1), old_value, new_value)
        fill_recursively(array_of_pixels, (row, col - 1), old_value, new_value)


def fill_region(array_of_pixels, position, new_value):
    """ fills a region of a 2D numpy array with the same value

    :type array_of_pixels: np.ndarray
    :param array_of_pixels: 2D numpy array of all pixel values
    :param position: tuple with (row, col) location of pixel in the region to modify
    :type position: tuple
    :type new_value: float
    :param new_value: new value for pixel at `position` and all pixels in same region
    :return: None
    """
    row, col = position
    old_value = array_of_pixels[row, col]
    fill_recursively(array_of_pixels, position, old_value, new_value)


# constant value that the image array is padded with
PAD_VALUE = -1


def pad_with(vector, pad_width, iaxis, kwargs):
    """helper function that is called by np.pad to surround a nparray with a constant value
    Example: [[0,0],[0,0]] becomes [[-1,-1,-1, -1],[-1, 0, 0, -1],[-1, 0, 0, -1],[-1,-1,-1, -1]]
    """
    pad_value = kwargs.get('padder', PAD_VALUE)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
