# Helper Functions: Optimal Arc When Two Points Are Known
# https://arxiv.org/pdf/1504.06582.pdf
# All formulas are from page 9 of the above document
# These functions are called in constrained_circle_fit.py

import math


def M(g, h, x, y):
    n = len(x)  # should also be len(y)
    summation = 0
    for xi, yi in zip(x, y):
        summation += xi ** g * yi ** h
    return summation / n


def qxy(point_a, x, y):
    xa, ya = point_a
    return 2 * (xa * ya - (xa * M(0, 1, x, y) + ya * M(1, 0, x, y)) + M(1, 1, x, y))


def qxx(point_a, x, y):
    xa = point_a[0]
    return xa ** 2 - 2 * xa * M(1, 0, x, y) + M(2, 0, x, y)


def qyy(point_a, x, y):
    ya = point_a[1]
    return ya ** 2 - 2 * ya * M(0, 1, x, y) + M(0, 2, x, y)


def qy(point_a, x, y):
    xa, ya = point_a
    return -ya ** 3 + (xa ** 2 + ya ** 2) * M(0, 1, x, y) + ya * ((M(2, 0, x, y) + M(0, 2, x, y)) - xa ** 2) - (
            M(0, 3, x, y) + M(2, 1, x, y))


def qx(point_a, x, y):
    xa, ya = point_a
    return -xa ** 3 + (xa ** 2 + ya ** 2) * M(1, 0, x, y) + xa * ((M(2, 0, x, y) + M(0, 2, x, y)) - ya ** 2) - (
            M(3, 0, x, y) + M(1, 2, x, y))


def q(point_a, x, y):
    xa, ya = point_a
    return (1 / 4) * ((xa ** 2 + ya ** 2) ** 2 - 2 * (xa ** 2 + ya ** 2) * (M(2, 0, x, y) + M(0, 2, x, y)) + (
            M(4, 0, x, y) + 2 * M(2, 2, x, y) + M(0, 4,
                                                  x, y)))


def b2():
    return 1


def b1(point_a, point_p, alpha, beta):
    xa, ya = point_a
    xp, yp = point_p
    return -2 * (alpha * xa + beta * ya) + 2 * (xp * alpha + yp * beta)


def b0(point_a, point_p):
    xa, ya = point_a
    xp, yp = point_p
    return (xa ** 2 + ya ** 2) - 2 * (xp * xa + yp * ya) + (xp ** 2 + yp ** 2)


def a2(point_a, alpha, beta, x, y):
    return alpha ** 2 * qxx(point_a, x, y) + alpha * beta * qxy(point_a, x, y) + beta ** 2 * qyy(point_a, x, y)


def a1(point_a, point_p, alpha, beta, x, y):
    xp, yp = point_p
    return alpha * qx(point_a, x, y) + beta * qy(point_a, x, y) + 2 * (
            xp * alpha * qxx(point_a, x, y) + yp * beta * qyy(point_a, x,
                                                              y)) + (
                   xp * beta + yp * alpha) * qxy(point_a, x, y)


def a0(point_a, point_p, x, y):
    xp, yp = point_p
    return q(point_a, x, y) + xp * qx(point_a, x, y) + yp * qy(point_a, x, y) + xp ** 2 * qxx(point_a, x,
                                                                                              y) + xp * yp * qxy(
        point_a, x,
        y) + yp ** 2 \
           * qyy(point_a, x, y)


# Function to calculate distance
def distance(point_1, point_2):
    """ calculates the euclidian distance between two points

    :param point_1:
    :param point_2:
    :return:
    """
    x1 = point_1[0]
    x2 = point_2[0]
    y1 = point_1[1]
    y2 = point_2[1]
    # Calculating distance
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
