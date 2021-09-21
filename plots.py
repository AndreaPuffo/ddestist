import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt


def plot_all_ist(X, Y):
    colors = ['b', 'r', 'g', 'y', 'k', 'm']
    color_plots = [colors[int(Y[p]%len(colors))] for p in range(len(Y))]

    plot_range = min(len(Y), 1000)
    for p in range(plot_range):
        plt.scatter(X[p,0] * np.cos(X[p, 1]), X[p,0] * np.sin(X[p, 1]),
                    color=color_plots[p])  # , label=f'{Y[p]+1} h')

    # plt.legend()
    plt.grid()
    # plt.show()


def trace_plots(trace, net, varnames):
    for name in varnames:
        pm.traceplot(trace[-1000:], var_names=[name],
                     lines=[(name, {}, list(net.fc1.bias.data.numpy()))])

    # pm.traceplot(trace[-1000:], var_names=['w_in_1'],
    #              lines=[('w_in_1', {}, list(net.fc1.weight.data.numpy()))])


def plot_prob_surface(ppc, grid, X_test, Y_test, tau_ist):
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
    colors = ['b', 'r', 'g', 'y', 'k', 'm']
    fig, ax = plt.subplots(figsize=(16, 9))
    contour = ax.contourf(grid[0], grid[1],
                          ppc.mean(axis=0)[:, tau_ist].reshape(100, 100), cmap=cmap)
    n_tau = np.max(Y_test)
    for tau in range(n_tau):
        # ax.scatter(X_test[Y_test == 0, 0], X_test[Y_test == 0, 1])
        ax.scatter(X_test[Y_test == tau, 0], X_test[Y_test == tau, 1], color=colors[tau%len(colors)])
        # ax.scatter(X_test[Y_test == 2, 0], X_test[Y_test == 2, 1], color="g")
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(0.0, 3.5), ylim=(-3.2, 3.2), xlabel=r"$\rho$", ylabel=r"$\phi$")
    cbar.ax.set_ylabel("Posterior predictive mean probability of class label = {}".format(tau_ist))


def plot_uncertainty(ppc, grid, X_test, Y_test):
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    colors = ['b', 'r', 'g', 'y', 'k', 'm']

    fig, ax = plt.subplots(figsize=(16, 9))
    contour = ax.contourf(grid[0], grid[1],
                          ppc.std(axis=0)[:, 0].reshape(100, 100), cmap=cmap)
    # contour = ax.contourf(grid[0], grid[1],
    #                       ppc.std(axis=0)[:, 1].reshape(100, 100), cmap=cmap)
    n_tau = np.max(Y_test)
    for tau in range(n_tau):
        # ax.scatter(X_test[Y_test == 0, 0], X_test[pred == 0, 1])
        ax.scatter(X_test[Y_test == tau, 0], X_test[Y_test == tau, 1], color=colors[tau%len(colors)])
        # ax.scatter(X_test[Y_test == 2, 0], X_test[pred == 2, 1], color="g")
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(0, 3.5), ylim=(-3.2, 3.2), xlabel="X", ylabel="Y")
    cbar.ax.set_ylabel(f"Uncertainty (posterior predictive standard deviation)")
