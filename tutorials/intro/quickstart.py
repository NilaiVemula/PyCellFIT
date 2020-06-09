#!/usr/bin/env python3

import matplotlib.pyplot as plt

from pycellfit.utils import read_segmented_image


def main():
    filename = 'hex.tif'

    # load input image as a numpy ndarray
    img_array = read_segmented_image(filename, visualize=True)
    plt.show()

    # print(img_array)

    #


if __name__ == '__main__':
    main()
