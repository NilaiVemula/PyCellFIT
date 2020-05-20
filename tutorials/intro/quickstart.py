#!/usr/bin/env python3

import matplotlib.pyplot as plt

from pycellfit.utils import read_segmented_image


def main():
    filename = 'hex.tif'

    imgarray = read_segmented_image(filename, visualize=True)
    plt.show()

    print(imgarray)


if __name__ == '__main__':
    main()
