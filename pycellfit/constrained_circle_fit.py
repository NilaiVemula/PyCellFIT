# Optimal Arc When Two Points Are Known
# https://arxiv.org/pdf/1504.06582.pdf

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from pycellfit.circle_fit_helpers import a0, a1, a2, b0, b1, b2, distance


def r(point_a, point_p, alpha, beta, t):
    """ calculate radius of circle

    :param point_a:
    :param point_p:
    :param alpha:
    :param beta:
    :param t:
    :return:
    """
    xa, ya = point_a
    xp, yp = point_p

    return math.sqrt((xa - (xp + alpha * t)) ** 2 + (ya - (yp + beta * t)) ** 2)


def f(ans, x, y, point_a, point_p):
    """ cost function that needs to be minimized
    See https://arxiv.org/pdf/1504.06582.pdf (page 9)

    :param ans:
    :param x:
    :param y:
    :param point_a:
    :param point_p:
    :return:
    """
    t, alpha, beta = ans

    numerator = (a0(point_a, point_p, x, y) + a1(point_a, point_p, alpha, beta, x, y) * t + a2(point_a,
                                                                                               alpha, beta,
                                                                                               x,
                                                                                               y) * t ** 2)
    denominator = b0(point_a, point_p) + b1(point_a, point_p, alpha, beta) * t + b2() * t ** 2
    return numerator / denominator


def t_alpha_beta_to_center_radius(ans, point_a, point_p):
    """ converts from parametric form (t, alpha, beta) to standard form (radius, center)

    :param ans:
    :param point_a:
    :param point_p:
    :return:
    """
    t, alpha, beta = ans

    xp, yp = point_p

    xc = xp + alpha * t
    yc = yp + beta * t

    radius = r(point_a, point_p, alpha, beta, t)

    return xc, yc, radius


def center_point_to_t_alpha_beta(center, point_p):
    """ converts from standard form (center, point on circle) to parametric form (t, alpha, beta)

    :param center:
    :param point_p:
    :return:
    """
    xp, yp = point_p
    xc, yc = center

    # solved this in mathematica
    t = (-xc + xp) / math.sqrt(
        (xc ** 2 - 2 * xc * xp + xp ** 2) / (xc ** 2 - 2 * xc * xp + xp ** 2 + yc ** 2 - 2 * yc * yp + yp ** 2))
    alpha = -math.sqrt(
        ((xc ** 2 - 2 * xc * xp + xp ** 2) / (xc ** 2 - 2 * xc * xp + xp ** 2 + yc ** 2 - 2 * yc * yp + yp ** 2)))
    beta = -(((-yc + yp) * (math.sqrt(
        (xc ** 2 - 2 * xc * xp + xp ** 2) / (xc ** 2 - 2 * xc * xp + xp ** 2 + yc ** 2 - 2 * yc * yp + yp ** 2)))) / (
                     -xc + xp))

    return t, alpha, beta


def algebraic_circle_fit(point_1, point_2, point_3):
    """ finds center and radius of a circle that contains three points on its edge

    :param point_1:
    :param point_2:
    :param point_3:
    :return:
    """
    # Function to find the circle on
    # which the given three points lie

    x1 = point_1[0]
    x2 = point_2[0]
    x3 = point_3[0]

    y1 = point_1[1]
    y2 = point_2[1]
    y3 = point_3[1]

    c = (x1 - x2) ** 2 + (y1 - y2) ** 2
    a = (x2 - x3) ** 2 + (y2 - y3) ** 2
    b = (x3 - x1) ** 2 + (y3 - y1) ** 2
    s = 2 * (a * b + b * c + c * a) - (a * a + b * b + c * c)
    xc = (a * (b + c - a) * x1 + b * (c + a - b) * x2 + c * (a + b - c) * x3) / s
    yc = (a * (b + c - a) * y1 + b * (c + a - b) * y2 + c * (a + b - c) * y3) / s
    ar = a ** 0.5
    br = b ** 0.5
    cr = c ** 0.5
    radius = ar * br * cr / ((ar + br + cr) * (-ar + br + cr) * (ar - br + cr) * (ar + br - cr)) ** 0.5

    return xc, yc, radius


def fit(x, y, start_point, end_point):
    """ performs a constrained circle fit based on points around an arc and the start and end points of the arc

    :param x:
    :param y:
    :param start_point:
    :param end_point:
    :return:
    """
    point_a, point_p = start_point, end_point

    # Run Algebraic fit with three points to serve as initial guess
    # choose random point in list and use start and end points
    index = random.randint(0, len(x) - 1)
    random_intermediate_point = (x[index], y[index])
    print("intermediate point: " + str(random_intermediate_point))
    guess = algebraic_circle_fit(start_point, random_intermediate_point, end_point)
    print(guess)
    center_estimate = (guess[0], guess[1])
    ti, alphai, betai = center_point_to_t_alpha_beta(center_estimate, point_p)

    # t, alpha, beta
    estimate = np.array([ti, alphai, betai])
    print("ESTIMATE:" + str(estimate))

    cons = [{'type': 'eq', 'fun': constraint}, {'type': 'eq', 'fun': constraint2, 'args': (point_a, point_p)}]
    # noinspection PyTypeChecker
    ans = optimize.minimize(fun=f, x0=estimate, args=(x, y, point_a, point_p), constraints=cons)
    ans = ans.x
    xc, yc, radius = t_alpha_beta_to_center_radius(ans, point_a, point_p)

    return xc, yc, radius


def constraint(ans):
    """ first constraint for the solver: alpha^2 + beta^2 = 1

    :param ans:
    :return:
    """
    t, alpha, beta = ans

    return alpha ** 2 + beta ** 2 - 1


def constraint2(ans, point_a, point_p):
    """ second constraint for the solver: distance from center to point_a = distance from center to point_p

    :param ans:
    :param point_a:
    :param point_p:
    :return:
    """
    xc, yc, radius = t_alpha_beta_to_center_radius(ans, point_a, point_p)
    center = (xc, yc)

    r_ca = distance(point_a, center)
    r_cp = distance(point_p, center)

    return r_ca - r_cp


def plot_data_and_circle_fit(x, y, xc, yc, radius, start_point, end_point):
    """plots the results of the circular fit

    :param x: nparray of all x coordinates of data
    :param y: nparray of all y coordinates of data
    :param xc: float of x coordinate of center of fit circle
    :param yc: float of y coordinate of center of fit circle
    :param radius: radius of fit circle
    :param start_point:
    :param end_point:
    :return: None
    """
    plt.figure(facecolor='white')
    plt.axis('equal')

    # find the initial and final theta that the arc spans
    start_theta = math.atan2(start_point[1] - yc, start_point[0] - xc)
    end_theta = math.atan2(end_point[1] - yc, end_point[0] - xc)
    # theta_fit = np.linspace(start_theta, end_theta, 180)
    theta_fit = np.linspace(math.pi / 2 - 0.3, math.pi + 0.3, 180)
    # stores all x and y coordinates along the fitted arc
    x_fit = xc + radius * np.cos(theta_fit)
    y_fit = yc + radius * np.sin(theta_fit)

    # plot least squares circular arc
    plt.plot(x_fit, y_fit, 'b-', label="fitted circle", lw=2)
    # plot center
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    # plot start_point
    plt.plot([start_point[0]], [start_point[1]], 'rD', mec='y', mew=1)
    # plot end_point
    plt.plot([end_point[0]], [end_point[1]], 'rD', mec='y', mew=1)
    # plot data
    plt.plot(x, y, 'r.', label='data', mew=1)

    # plot formatting
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best', labelspacing=0.1)
    plt.title('Constrained Fit')
    plt.grid()


def test1():
    # first quadrant
    x = np.array([0.5, math.sqrt(2) / 2, math.sqrt(3) / 2])
    y = np.array([math.sqrt(3) / 2, math.sqrt(2) / 2, 0.5])
    start_point = (0, 1)
    end_point = (1, 0)

    xc, yc, radius = fit(x, y, start_point, end_point)
    print("xc: " + str(xc))
    print("yc: " + str(yc))
    print("radius: " + str(radius))

    center = (xc, yc)

    print("VERIFICATION:")
    print(distance(center, start_point), distance(center, end_point))

    plot_data_and_circle_fit(x, y, xc, yc, radius, start_point, end_point)
    # show plot
    plt.show()


def test2():
    x = np.array([-math.sqrt(3.9) / 2, -math.sqrt(2.1) / 2, -0.55])
    y = np.array([0.55, math.sqrt(2.6) / 2, (0.1 + math.sqrt(3)) / 2])
    start_point = (-1, 0)
    end_point = (0, 1)

    xc, yc, radius = fit(x, y, start_point, end_point)
    print("xc: " + str(xc))
    print("yc: " + str(yc))
    print("radius: " + str(radius))

    center = (xc, yc)

    print("VERIFICATION:")
    print(distance(center, start_point), distance(center, end_point))

    plot_data_and_circle_fit(x, y, xc, yc, radius, start_point, end_point)
    # show plot
    plt.show()


if __name__ == '__main__':
    test2()
