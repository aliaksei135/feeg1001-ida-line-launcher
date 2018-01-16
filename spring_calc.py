import numpy as np
import matplotlib.pyplot as pl
from numpy.core.umath import pi, cos, sin

def spherecd(Re):
    cd = (24 / Re) + (2.6 * (Re / 5)) / (1 + (Re / 5) ** 1.52) + (0.411 * (Re / 263000) ** -7.94) / (
        1 + (Re / 263000) ** -8) + (Re ** 0.8 / 461000)
    return cd

rho = 1.225  # density of air
nu = 1.48e-5  # kinematic viscosity of air
g = 9.81  # acceleration due to gravity
d = 0.04  # diameter of sphere
S = pi * (d / 2) ** 2  # frontal area of sphere
m = 0.024  # mass of sphere
mr = 0.05 # mass of internal launch rig



def do_calc(k, init_x, alpha, do_plot=False):
    assert isinstance(init_x, np.float64)

    deltaT = 0.00001  # We'll need a much smaller time step
    l = np.round(1 / deltaT)

    v = np.zeros(np.int(l))  # velocity
    Re = np.zeros(np.int(l))  # Reynolds number
    cd = np.zeros(np.int(l))  # drag coefficient
    f = np.zeros(np.int(l))  # spring force
    x = np.zeros(np.int(l))  # spring length

    x[0] = np.float(init_x)
    v[0] = 0.00001  # need to use a very small initial velocity to avoid divide by zeros
    Re[0] = v[0] * d / nu  # initial Reynolds number
    cd[0] = spherecd(Re[0])  # initial drag coefficient

    i = 0
    while (x[i] > 0):
        fs = x[i] * k  # spring force
        fd = -0.5 * rho * v[i] ** 2 * S * cd[i]  # drag force
        fg = (9.81 * m) / cos(alpha) # force due to gravity
        fr = -((mr * g) * sin(alpha))
        f[i] = fs + fd + fg + fr # total force
        a = (fs + fd + fg + fr) / m  # acceleration
        v[i + 1] = v[i] + a * deltaT  # new velocity
        x[i + 1] = x[i] - v[i] * deltaT  # new position
        Re[i] = v[i + 1] * S / nu  # Reynolds number
        cd[i + 1] = spherecd(Re[i])  # drag coefficient
        i += 1

    if do_plot:
        print('Exit Velocity', v[i])
        pl.plot(x[0:i], v[0:i])
        pl.xlabel('Spring compression [m]')
        pl.ylabel('velocity [m]')
        pl.show()

    return v[i]


if __name__ == "__main__":
    do_calc(1310, np.float64(0.05), 50, True)
