from multiprocessing import Pool

import matplotlib.pyplot as pl
import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.plotly as py

import spring_calc
import test_data
import traj_calc


## GLOBAL CONSTS ##
# global k, \
#     startx, deltax, endx, \
#     startalpha, deltaalpha, endalpha, \
#     target_dist, target_height_at_dist, \
#     x_deflections, y_alphas, z_height_at_target


def find_nearest(flt_array, target, tol):
    return (np.abs(flt_array - target)).argmin()

def do_iter(x_count, x, ys, target_dist, k, init_height, grid_height):
    y_count = 0
    zs = []
    for y in ys:
        exit_v = spring_calc.do_calc(k, x, y)
        # only need h here but the rest of can come along for the ride as well
        vX, vY, x_dist, h = traj_calc.do_calc(exit_v, y, init_height)
        try:
            if max(x_dist) < target_dist:
                raise StopIteration
            else:
                h1, h2 = np.split(h, 2)
                target_dist_index = find_nearest(h2, grid_height, 0.05) + (len(h1) - 1)
                if x_dist[target_dist_index] < target_dist:
                    raise StopIteration
        except StopIteration:
            print('Target Distance not achieved')
            zs.append(0)
            continue
        zs.append(x_dist[target_dist_index])
        y_count += 1
    print('Completed iters for x={}'.format(x_count))
    return zs


def make_full_plot(multithread=True):
    # global k, \
    #     startx, deltax, endx, \
    #     startalpha, deltaalpha, endalpha, \
    #     target_dist, target_height_at_dist, \
    #     x_deflections, y_alphas, z_height_at_target
    #Constants
    k = 1310 #Spring const [Nm]
    startx = 0.04 # Min spring deflection
    deltax = 0.01 #Change in spring deflection per iteration
    endx = 0.101 #Max spring deflection
    startalpha = 20 # Min launch angle
    deltaalpha = 5 #Change in launch angle per iteration
    endalpha = 82 #Max launch angle
    target_dist = 5. #Distance between target and launcher
    init_height = 0.1 #Initial height of ball at point of launch
    grid_height = 0.45 #Height of target grid

    x_deflections = np.arange(startx, endx, deltax)
    y_alphas = np.deg2rad(np.arange(startalpha, endalpha, deltaalpha))
    z_dist = np.zeros((len(x_deflections), len(y_alphas)), dtype=np.float64)

    if(not multithread):
        # Probably inaccurate to the the point of being arbitrary but still looks cool :)
        print('Expected time to complete: {} seconds = {} minutes'.format(
            ((len(x_deflections) * len(y_alphas)) / 3.2), ((len(x_deflections) * len(y_alphas)) / 3.2) / 60))
        ## SINGLE THREAD ##
        i = 0
        x_count = 0
        miss_count = 0
        for x in x_deflections:
            y_count = 0
            for y in y_alphas:
                exit_v = spring_calc.do_calc(k, x, y)
                # only need h here but the rest of can come along for the ride as well
                vX, vY, x_dist, h = traj_calc.do_calc(exit_v, y, init_height)
                try:
                    if max(x_dist) < target_dist:
                        raise StopIteration
                    else:
                        h1, h2 = np.split(h, 2)
                        target_dist_index = find_nearest(h2, grid_height, 0.05) + (len(h1) - 1)
                        if x_dist[target_dist_index] < target_dist:
                            raise StopIteration
                except StopIteration:
                    print('Target Distance not achieved')
                    miss_count += 1
                    continue
                z_dist[x_count, y_count] = x_dist[target_dist_index]
                print('Completed iter', i+1)
                i += 1
                y_count += 1
            x_count += 1

        if miss_count > 50:
            print("Target Distance not achieved {} times. Optimise starting variables?".format(miss_count))
        ## /> ##
    else:
        # Probably inaccurate to the the point of being arbitrary but still looks cool :)
        print('[MULTITHREAD] Expected time to complete: {} seconds = {} minutes'.format(
            ((len(x_deflections) * len(y_alphas)) / 7.4), ((len(x_deflections) * len(y_alphas)) / 8.4) / 60))
        ## MULTITHREADED ##
        pool = Pool(8)
        res = [pool.apply_async(do_iter, (count, x_val, y_alphas[0:], target_dist, k, init_height, grid_height)) for count, x_val in enumerate(x_deflections)]
        for idx, zs in enumerate(res):
            z_dist[idx] = res[idx].get()
        ## /> ##


    X = x_deflections
    Y = np.rad2deg(y_alphas)
    Z = z_dist.T

    ## PLOTLY VERSION ##
    data = [
        go.Contour(x=X,
                   y=Y,
                   z=Z,
                   colorscale='heatmap',
                   contours=dict(
                       start=0,
                       end=np.floor(np.amax(z_dist)),
                       size=0.5,
                       showlabels=True,
                   ),
                   line=dict(
                       smoothing=0.8
                   ),
                   colorbar=dict(
                       title='Height at Target [m]'
                   )
                   )
    ]
    layout = go.Layout(
            title='Traj Calculation',
            autosize=True,
            xaxis=dict(
                title='Spring Deflection [m]',
                showgrid=True),
            yaxis=dict(
                title='Launch Angle [deg]',
                showgrid=True),
        )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)
    ## /> ##

    ## MATPLOTLIB VERSION ##
    # fig = pl.figure()
    # ax = fig.add_subplot(121, projection='3d')
    # ax = fig.gca(projection='3d')
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

    # SLICER
    # X_slice, Y_slice = [], []
    # for i in range(len(test_data.z)):
    #     for j in range(len(test_data.z[i])):
    #         if np.abs(test_data.z[i,j] - target_height_at_dist) < 0.1:
    #             X_slice.append(test_data.x[i])
    #             Y_slice.append(np.rad2deg(test_data.y[j]))
    #
    # pl.plot(X_slice, Y_slice, 'kx-')
    # pl.set_xlabel('Spring Deflection [m]')
    # pl.set_ylabel('Launch Angle [deg]')

    # pl.tight_layout()
    # pl.show()
    ## /> ##

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

    p = np.polyfit(X_slice, Y_slice, 5)
    x_fit = np.arange(0.03, 0.1, 0.001)
    y_fit = np.polyval(p, x_fit)

    pl.plot(X_slice, Y_slice, 'kx')
    pl.plot(x_fit, y_fit, 'r-')

    pl.xlim(0.03, 0.1)
    pl.ylim(0, 90)

    pl.tight_layout()
    pl.show()

if __name__ == "__main__":
    plotly.tools.set_credentials_file(username='aliaksei135', api_key='gfEwwfj8ox9UkeaZPONl')
    plotly.tools.set_config_file(world_readable=True)
    make_full_plot()
    # test_slice()