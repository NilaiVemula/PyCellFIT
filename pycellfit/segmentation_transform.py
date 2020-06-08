""" functions to convert between watershed and skeleton segmented images"""

import numpy as np
from scipy import stats

from pycellfit.segmentation_transform_utils import fill_region, pad_with, PAD_VALUE


def skeleton_to_watershed(skeleton_image_array, region_value=0, boundary_value=225):
    """ converts a segmented skeleton image (all regions are same value with region boundaries being a second value)
    to a watershed segmented image (each region has a unique value and there are no boundary pixels, background
    region has value of zero)

    :param skeleton_image_array: 2D numpy array with pixel values of a skeleton segmented image
    :type skeleton_image_array: np.ndarray
    :param region_value: value of pixels in regions in skeleton_segmented images (default is 0)
    :type region_value: float
    :param boundary_value: value of boundary pixels of regions in skeleton_segmented images (default is 255)
    :type boundary_value: float
    :return: watershed_image_array
    :rtype watershed_image_array: np.ndarray
    """

    # throw error if skeleton_image_array is not two dimensions
    if np.ndim(skeleton_image_array) != 2:
        raise ValueError()

    # throw error if skeleton_image_array is not np.ndarray
    if not isinstance(skeleton_image_array, np.ndarray):
        raise TypeError()

    # copy input array (skeleton image)
    watershed_image_array = np.copy(skeleton_image_array)

    # fill each region with unique value
    new_value = 1
    with np.nditer(watershed_image_array, flags=['multi_index'], op_flags=['readwrite']) as iterator:
        for old_value in iterator:
            if new_value == boundary_value:
                # make sure we don't label any regions with same value as boundary value
                new_value += 1
            if old_value == boundary_value:
                # don't touch boundary pixels
                break
            elif old_value == region_value:
                # if pixel has not been touched, fill region
                position = iterator.multi_index
                fill_region(watershed_image_array, position, new_value)
                new_value += 1

    # determine pixel value of background
    # potential background values are the values in the four corners of the array
    potential_background_values = watershed_image_array[[0, 0, -1, -1], [0, -1, 0, -1]]
    # we determine the background value to be the mode of the potential values
    background_value = stats.mode(potential_background_values)[0][0]

    # search for position of background value
    search_results = np.where(watershed_image_array == background_value)
    background_position = list(zip(search_results[0], search_results[1]))[0]

    # fill background region with zeros
    fill_region(watershed_image_array, background_position, new_value=0)

    # TODO: remove boundaries

    return watershed_image_array


def watershed_to_skeleton(watershed_image_array, region_value=0, boundary_value=255):
    """

    :param watershed_image_array:
    :param region_value:
    :param boundary_value:
    :return:
    """

    # create a one pixel wide padding around the edge of the image for easy iteration
    padded_array = np.pad(watershed_image_array, 1, pad_with)
    # print(padded_array.shape)

    # initialize a new nparray to hold skeleton segmented image (mask)
    skeleton = np.full(watershed_image_array.shape, region_value)

    # if a pixel neighbors a pixel of a different value, consider it a boundary point and add it to the skeleton image
    for row in range(1, padded_array.shape[0] - 1):
        for col in range(1, padded_array.shape[1] - 1):

            neighboring_values = [padded_array[row, col],
                                  padded_array[row, col + 1],
                                  padded_array[row + 1, col],
                                  padded_array[row + 1, col + 1]]

            # remove duplicates
            neighboring_values = list(set(neighboring_values))

            # remove padded value if in list of neighboring values
            try:
                neighboring_values.remove(PAD_VALUE)
            except ValueError:
                pass

            if len(neighboring_values) >= 2:
                skeleton[row - 1, col - 1] = boundary_value
    # FIXME: right most column and bottom row have boundary_value when they should not be
    return skeleton
