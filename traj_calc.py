import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize

# constants
from numpy.core.umath import pi, cos, sin, sqrt


def spherecd(Re):
    cd = (24 / Re) + (2.6 * (Re / 5)) / (1 + (Re / 5) ** 1.52) + (0.411 * (Re / 263000) ** -7.94) / (
        1 + (Re / 263000) ** -8) + (Re ** 0.8 / 461000)
    return cd


##Env constants
rho = 1.225  # density of air
nu = 1.48e-5  # kinematic viscosity of air
g = 9.81  # acceleration due to gravity
d = 0.04  # diameter of sphere
S = pi * (d / 2) ** 2  # frontal area of sphere
m = 0.015  # mass of sphere
deltaT = 0.0001  # time step for our simulation


def do_calc(v, alpha, do_plot=False, stop_at=7):
    # initialize arrays
    # we expect the sphere to be in the air no more than three seconds
    # so set our array to be of length 3/deltaT (this is quite crude, but works!)
    l = round(5 / deltaT)
    # now initialize arrays for quantities of interest
    h = np.zeros(l)  # height of sphere
    x = np.zeros(l)  # horizontal ordinate of sphere
    vX = np.zeros(l)  # horizontal velocity
    vY = np.zeros(l)  # vertical velocity
    Re = np.zeros(l)  # Reynolds number
    cd = np.zeros(l)  # drag coefficient

    # starting conditions
    h[0] = 0.1  # height of sphere at start
    x[0] = 0  # horizontal ordinate of sphere at start
    vX[0] = v * cos(alpha)  # initial horizontal velocity
    vY[0] = v * sin(alpha)  # initial vertical velocity
    Re[0] = sqrt(vX[0] ** 2 + vY[0] ** 2) * d / nu  # initial Reynolds number
    cd[0] = spherecd(Re[0])  # initial drag coefficient

    i = 0
    while (h[i] > 0) and i + 1 < l:
        vY[i + 1] = vY[i] - (0.5 * rho * vY[i] ** 2 * S * cd[i] / m) * deltaT  # vel. change due to vertical drag
        vX[i + 1] = vX[i] - (0.5 * rho * vX[i] ** 2 * S * cd[i] / m) * deltaT  # vel. change due to horizontal drag
        vY[i + 1] = vY[i + 1] - g * deltaT  # vel. change due to gravity
        x[i + 1] = x[i] + vX[i + 1] * deltaT  # new horizontal position
        h[i + 1] = h[i] + vY[i + 1] * deltaT  # new vertical position
        Re[i + 1] = sqrt(vX[i + 1] ** 2 + vY[i + 1] ** 2) * d / nu  # Reynolds number
        cd[i + 1] = spherecd(Re[i + 1])  # drag coefficient
        i = i + 1
        # if x[i] > stop_at :
        #     break

    if do_plot:
        pl.plot(x[0:i], h[0:i])
        pl.xlabel('distance [m]')
        pl.ylabel('height [m]')
        pl.show()

    return vX, vY, x, h


def get_max_height(v, alpha):
    vX, vY, x, h = do_calc(v, alpha)
    # print('Max height: ', max(h))
    return max(h)

if __name__ == "__main__":
    do_calc(30.2554897544, np.deg2rad(50), True)
