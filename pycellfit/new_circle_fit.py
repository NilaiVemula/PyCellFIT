import math

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy import optimize


def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def f(center, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """

    xc, yc = center
    Ri = calc_R(x, y, xc, yc)
    return np.sum((Ri - Ri.mean()) ** 2)


def distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def constraint2(center, point_a, point_p):
    """ second constraint for the solver: distance from center to point_a = distance from center to point_p

    :param ans:
    :param point_a:
    :param point_p:
    :return:
    """
    r_ca = distance(point_a, center)
    r_cp = distance(point_p, center)

    return r_ca - r_cp


def constraint1(center, point_a, point_p, x, y):
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()

    r_ca = distance(point_a, center)
    r_cp = distance(point_p, center)

    return np.abs(R - r_cp) + np.abs(R - r_ca)


def fit(x, y, start_point, end_point):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    # center, ier = optimize.leastsq(f, center_estimate, args=(x,y))

    # cons = [{'type': 'eq', 'fun': constraint1, 'args': (start_point, end_point, x, y)}, {'type': 'eq',
    #                                                                                      'fun': constraint2,
    #                                                                                      'args': (
    #                                                                                      start_point, end_point)}]
    # # noinspection PyTypeChecker
    # center = optimize.minimize(fun=f, x0=center_estimate, args=(x, y), constraints=cons)
    center = optimize.minimize(fun=f, x0=center_estimate, args=(x, y))
    if not center.success:
        print(center)
    center = center.x

    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R) ** 2)
    return xc, yc, R


def plot_data_circle(x, y, xc, yc, R):
    f = plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
    plt.axis('equal')

    theta_fit = np.linspace(-math.pi, math.pi, 180)

    x_fit = xc + R * np.cos(theta_fit)
    y_fit = yc + R * np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'b-', label="fitted circle", lw=2)
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    plt.xlabel('x')
    plt.ylabel('y')
    # plot data
    plt.plot(x, y, 'r-.', label='data', mew=1)

    plt.legend(loc='best', labelspacing=0.1)
    plt.grid()
    plt.title('Least Squares Circle')


def test1():
    # first quadrant
    x = np.array([0, 0.5, math.sqrt(2) / 2, math.sqrt(3) / 2, 1])
    y = np.array([1, math.sqrt(3) / 2, math.sqrt(2) / 2, 0.5, 0])
    start_point = (0, 1)
    end_point = (1, 0)

    xc, yc, radius = fit(x, y, start_point, end_point)
    print("xc: " + str(xc))
    print("yc: " + str(yc))
    print("radius: " + str(radius))

    center = (xc, yc)

    print("VERIFICATION:")
    print(distance(center, start_point), distance(center, end_point))

    # plot_data_and_circle_fit(x, y, xc, yc, radius, start_point, end_point)
    # show plot
    # plt.show()


def test2():
    x = np.array([-1, -math.sqrt(3.9) / 2, -math.sqrt(2.1) / 2, -0.550, 0])
    y = np.array([0, 0.55, math.sqrt(2.6) / 2, (0.1 + math.sqrt(3)) / 2, 1])
    start_point = (-1, 0)
    end_point = (0, 1)

    # x = np.array([196.5, 204.5, 211.5, 219.5])
    # y = np.array([370.5, 374.5, 379.5, 383.5])
    # start_point = (196.5, 370.5)
    # end_point = (219.5, 383.5)
    x = np.array([266.5, 273.83333333, 282.16666667, 289.5])
    y = np.array([141.5, 136.5, 132.5, 127.5])
    start_point = (266.5, 141.5)
    end_point = (289.5, 127.5)
    xc, yc, radius = fit(x, y, start_point, end_point)
    print("xc: " + str(xc))
    print("yc: " + str(yc))
    print("radius: " + str(radius))

    center = (xc, yc)

    print("VERIFICATION:")
    print(distance(center, start_point), distance(center, end_point))
    start_theta = math.degrees(math.atan2(start_point[1] - yc, start_point[0] - xc))
    end_theta = math.degrees(math.atan2(end_point[1] - yc, end_point[0] - xc))
    # plot
    fig, ax = plt.subplots()
    e1 = patches.Arc((xc, yc), width=radius * 2, height=radius * 2, theta1=start_theta, theta2=end_theta)

    ax.add_patch(e1)
    plt.xlim(195, 5)
    plt.ylim(-5, 5)
    # plt.show()


if __name__ == '__main__':
    test2()
