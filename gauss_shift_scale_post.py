from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class
from RRAEs.utilities import get_data
import matplotlib.pyplot as plt
import pdb
import numpy as np
import os


def draw_grid_squares(params, plot_square=True, col=False):
    params = params.T
    x, y = params[0], params[1]
    N = int(np.sqrt(len(x)))  # Assuming the grid is square

    fig, ax = plt.subplots()
    if plot_square:
        for i in range(N - 1):
            for j in range(N - 1):
                # Get the corners of the square
                square_x = [
                    x[i * N + j],
                    x[i * N + j + 1],
                    x[(i + 1) * N + j + 1],
                    x[(i + 1) * N + j],
                    x[i * N + j],
                ]
                square_y = [
                    y[i * N + j],
                    y[i * N + j + 1],
                    y[(i + 1) * N + j + 1],
                    y[(i + 1) * N + j],
                    y[i * N + j],
                ]
                ax.plot(square_x, square_y, "r-")
        ax.scatter(x, y, color="b")
    else:
        if not col:
            colors = plt.cm.viridis(np.linspace(0, 1, N))
            for i in range(N):
                ax.scatter(
                    x[i * N : (i + 1) * N], y[i * N : (i + 1) * N], color=colors[i]
                )
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, N))
            for j in range(N):
                ax.scatter(x[j::N], y[j::N], color=colors[j])
    plt.show()


def draw_transformed_quadrilaterals(
    params, transformed_params, plot_square=True, col=False
):
    params = params.T
    x, y = params[0], params[1]
    tx, ty = transformed_params[0], transformed_params[1]
    N = int(np.sqrt(len(x)))  # Assuming the grid is square

    fig, ax = plt.subplots()
    if plot_square:
        for i in range(N - 1):
            for j in range(N - 1):
                # Get the corners of the transformed quadrilateral
                transformed_square_x = [
                    tx[i * N + j],
                    tx[i * N + j + 1],
                    tx[(i + 1) * N + j + 1],
                    tx[(i + 1) * N + j],
                    tx[i * N + j],
                ]
                transformed_square_y = [
                    ty[i * N + j],
                    ty[i * N + j + 1],
                    ty[(i + 1) * N + j + 1],
                    ty[(i + 1) * N + j],
                    ty[i * N + j],
                ]
                ax.plot(transformed_square_x, transformed_square_y, "r-")
        ax.scatter(tx, ty, color="green")
    else:
        if not col:
            colors = plt.cm.viridis(np.linspace(0, 1, N))
            for i in range(N):
                ax.scatter(
                    tx[i * N : (i + 1) * N], ty[i * N : (i + 1) * N], color=colors[i]
                )
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, N))
            for j in range(N):
                ax.scatter(tx[j::N], ty[j::N], color=colors[j])
    plt.show()


if __name__ == "__main__":
    problem = "2d_gaussian_shift_scale"
    method = "Strong"
    google = 400
    folder = f"{problem}/{method}_{problem}/"
    file = f"{method}_{problem}_{google}.pkl"
    trainor = RRAE_Trainor_class()  # RRAE_Trainor_class
    
    trainor.load_model("2d_gaussian_shift_scale/Strong_2d_gaussian_shift_scale_gen_nomax/Strong_2d_gaussian_shift_scale_14_gen.pkl")
    x_train, x_test, p_train, p_test = get_data(problem, google=google)[:4]

    draw_grid_squares(p_train)
    draw_grid_squares(p_train, plot_square=False, col=True)
    draw_grid_squares(p_train, plot_square=False, col=False)

    # PLOTTING LAT FIGS
    try:
        trans_p = trainor.model.latent(
            x_train, apply_basis=trainor.basis, get_coeffs=True
        )
    except AttributeError:
        trans_p = np.array(trainor.model.latent(x_train))[[0, 10]]

    draw_transformed_quadrilaterals(p_train, trans_p)
    draw_transformed_quadrilaterals(p_train, trans_p, plot_square=False, col=True)
    draw_transformed_quadrilaterals(p_train, trans_p, plot_square=False, col=False)

    # PLOTTING TEST SIMULATION
    try:
        pr = trainor.model(x_test, apply_basis=trainor.basis)
    except AttributeError:
        pr = trainor.model(x_test)

    # T, N = pr.shape
    # y_min = min(pr.min(), x_test.min())
    # y_max = max(pr.max(), x_test.max())

    # for i in range(N):
    #     plt.plot(pr[:, i], label=f'pred_{i}')
    #     plt.plot(x_test[:, i], label=f'x_test_{i}', linestyle='--')
    #     plt.show(block=False)
    #     plt.ylim([y_min, y_max])
    #     plt.pause(0.01)
    #     plt.clf()

    pdb.set_trace()
