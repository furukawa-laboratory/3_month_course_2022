import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

STEP = 150


def visualize_history(X, Y_history, Z_history, error_history, save_gif=False, filename="tmp"):
    input_dim, latent_dim = X.shape[1], Z_history[0].shape[1]
    input_projection_type = '3d' if input_dim > 2 else 'rectilinear'

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)
    input_ax = fig.add_subplot(gs[0:2, 0], projection=input_projection_type)
    latent_ax = fig.add_subplot(gs[0:2, 1], aspect='equal')
    error_ax = fig.add_subplot(gs[2, :])
    num_epoch = len(Y_history)

    if input_dim == 3 and latent_dim == 2:
        resolution = int(np.sqrt(Y_history.shape[1]))
        if Y_history.shape[1] == resolution ** 2:
            Y_history = np.array(Y_history).reshape((num_epoch, resolution, resolution, input_dim))

    observable_drawer = [None, None, draw_observable_2D,
                         draw_observable_3D][input_dim]
    latent_drawer = [None, draw_latent_1D, draw_latent_2D][latent_dim]

    ani = FuncAnimation(
        fig,
        update_graph,
        frames=num_epoch,  # // STEP,
        repeat=True,
        interval=50,
        fargs=(observable_drawer, latent_drawer, X, Y_history, Z_history, error_history, fig,
               input_ax, latent_ax, error_ax, num_epoch))
    plt.show()
    if save_gif:
        ani.save(f"{filename}.mp4", writer='ffmpeg')


def update_graph(epoch, observable_drawer, latent_drawer, X, Y_history,
                 Z_history, error_history, fig, input_ax, latent_ax, error_ax, num_epoch):
    fig.suptitle(f"epoch: {epoch}")
    input_ax.cla()
    #  input_ax.view_init(azim=(epoch * 400 / num_epoch), elev=30)
    latent_ax.cla()
    error_ax.cla()

    Y, Z= Y_history[epoch], Z_history[epoch]
    colormap = X[:, 0]
    print(X.shape)
    observable_drawer(input_ax, X, Y, colormap)
    latent_drawer(latent_ax, Z, colormap)
    draw_error(error_ax, error_history, epoch)


def draw_observable_3D(ax, X, Y, colormap):
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colormap)
    # ax.set_zlim(-1, 1)
    if len(Y.shape) == 3:
        ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')
        # ax.scatter(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')
    else:
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='black')
# ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='black')
# ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')


def draw_observable_2D(ax, X, Y, colormap):
    ax.scatter(X[:, 0], X[:, 1], c=colormap)
    ax.plot(Y[:, 0], Y[:, 1], c='black')


def draw_latent_2D(ax, Z, colormap):
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.scatter(Z[:, 0], Z[:, 1], c=colormap)


def draw_latent_1D(ax, Z, colormap):
    ax.scatter(Z, np.zeros(Z.shape), c=colormap)
    ax.set_ylim(-1, 1)

def draw_error(ax, error_history, epoch):
    ax.set_title("error_function", fontsize=8)
    ax.plot(error_history, label='誤差関数')
    ax.scatter(epoch, error_history[epoch], s=55, marker="*")