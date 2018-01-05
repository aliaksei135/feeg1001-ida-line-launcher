from struct import pack

from matplotlib.colors import Normalize

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import webbrowser

import spring_calc
import test_data
import traj_calc
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def make_calc(target_height, height_at_x):
    # Constants
    k = 500
    #Vars
    init_deflection = 0.1
    alpha = 45
    exit_v = spring_calc.do_calc(k, init_deflection, alpha)
    vX, vY, x, h = traj_calc.do_calc(exit_v, alpha)


def find_nearest(flt_array, target, tol):
    idx = (np.abs(flt_array - target)).argmin()
    # if np.abs(flt_array[idx] -target) > tol:
    #     raise StopIteration
    # else:
    #     return idx
    return idx


def make_full_plot():
    #Constants
    k = 1310 #Spring const [Nm]
    startx = 0.04 # Min spring deflection
    deltax = 0.005 #Change in spring deflection per iteration
    endx = 0.1 #Max spring deflection
    startalpha = 22 # Min launch angle
    deltaalpha = 2 #Change in launch angle per iteration
    endalpha = 86 #Max launch angle
    target_dist = 5. #Distance between target and launcher
    target_height_at_dist = 10. #Height ball should be at target_dist

    x_deflections = np.arange(startx, endx, deltax)
    y_alphas = np.deg2rad(np.arange(startalpha, endalpha, deltaalpha))
    z_height_at_target = np.zeros((len(x_deflections), len(y_alphas)), dtype=np.float64)

    # Probably inaccurate to the the point of being arbitrary but still looks cool :)
    print('Expected time to complete: {} seconds = {} minutes'.format(((len(x_deflections) * len(y_alphas))/3.2), ((len(x_deflections) * len(y_alphas))/3.2)/60))

    i = 0
    x_count = 0
    miss_count = 0
    for x in x_deflections:
        y_count = 0
        for y in y_alphas:
            exit_v = spring_calc.do_calc(k, x, y)
            # only need h here but the rest of can come along for the ride as well
            vX, vY, x_dist, h = traj_calc.do_calc(exit_v, y, stop_at=target_dist)
            try:
                if max(x_dist) < target_dist:
                    raise StopIteration
                else:
                    target_dist_index = find_nearest(x_dist, target_dist, 0.04)
            except StopIteration:
                print('Target Distance not achieved')
                miss_count += 1
                continue
            z_height_at_target[x_count, y_count] = h[target_dist_index]
            print('Completed iter', i+1)
            # print('{}, {}, {}'.format(exit_v, h[target_dist_index], np.rad2deg(y)))
            i += 1
            y_count += 1
        x_count += 1

    if miss_count > 50:
        print("Target Distance not achieved {} times. Optimise starting variables?".format(miss_count))

    # fig = pl.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(x_deflections, np.rad2deg(y_alphas), indexing='xy')
    Z = z_height_at_target.T

    data = [
        go.Surface(x=X, y=Y, z=Z)
    ]
    layout = go.Layout(
            title='Traj Calculation',
            autosize=True,
            scene=dict(
                xaxis=dict(title='Spring Deflection [m]'),
                yaxis=dict(title='Launch Angle [deg]'),
                zaxis=dict(title='Height at Target Distance [m]')
            )
        )
    fig = go.Figure(data=data, layout=layout)

    # ax.plot_surface(X, Y, Z, color='b', antialiased=True, linewidth=0, cmap='viridis')
    # ax.set_xlabel('Spring Deflection [m]')
    # ax.set_ylabel('Launch Angle [deg]')
    # ax.set_zlabel('Height at Target Distance [m]')
    # ax.set_zlim(0.5, 30.)

    # print('Generating 2D Slice...')
    # slice_ax = fig.add_subplot(122)
    # X_slice, Y_slice = [], []
    # flat_z_mins = np.abs(Z.flatten() - target_height_at_dist)
    # for idx, z in enumerate(flat_z_mins):
    #     if z < 0.8:
    #         y_idx = idx // len(x_deflections)
    #         x_idx = idx % len(x_deflections)
    #
    #         x_elem = startx + (x_idx * deltax)
    #         y_elem = startalpha + (y_idx * deltaalpha)
    #         X_slice.append(x_elem)
    #         Y_slice.append(y_elem)
    #
    # slice_ax.plot(X_slice, Y_slice, 'kx-')
    # slice_ax.set_xlabel('Spring Deflection [m]')
    # slice_ax.set_ylabel('Launch Angle [deg]')


    X_slice, Y_slice = [], []
    for i in range(len(test_data.z)):
        for j in range(len(test_data.z[i])):
            if np.abs(test_data.z[i,j] - target_height_at_dist) < 0.1:
                X_slice.append(test_data.x[i])
                Y_slice.append(np.rad2deg(test_data.y[j]))

    pl.plot(X_slice, Y_slice, 'kx-')
    # pl.set_xlabel('Spring Deflection [m]')
    # pl.set_ylabel('Launch Angle [deg]')

    # plot_url = py.plot(fig)

    pl.tight_layout()
    pl.show()



    # target_height_index = find_nearest(z_height_at_target, target_height_at_dist, 0.04)
    # x_target_index = target_height_index % len(y_alphas)
    # y_target_index = target_height_index % len(x_deflections)
    #
    # print('Target Height {} metres reached at {} metres away from launch with '
    #       'launch angle {} degrees and {} metres spring deflection'
    #       .format(target_height_at_dist, target_dist, y_alphas[y_target_index], x_deflections[x_target_index] ))

def test_slice():
    X_slice, Y_slice = [], []
    x, y, z = test_data.x, test_data.y, test_data.z
    for i in range(len(z)):
        for j in range(len(z[i])):
            if  np.abs(z[i, j] - 7.) < 0.08:
                a =  i % len(x)
                X_slice.append(x[i, j])
                Y_slice.append(y[i, j])
                break

    pl.plot(X_slice, Y_slice, 'kx-')
    # pl.set_xlabel('Spring Deflection [m]')
    # pl.set_ylabel('Launch Angle [deg]')

    # plot_url = py.plot(fig)

    pl.tight_layout()
    pl.show()

if __name__ == "__main__":
    # plotly.tools.set_credentials_file(username='aliaksei135', api_key='gfEwwfj8ox9UkeaZPONl')
    # plotly.tools.set_config_file(world_readable=True)
    # make_full_plot()
    test_slice()