import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

STEP = 150


def visualize_history(X, Y_history, U_history, V_history, error_history, animal_label, feature_label, save_gif=False, filename="tmp"):
    input_dim, latent_dim1, latent_dim2 = 1, U_history[0].shape[1], V_history[0].shape[1]
    input_projection_type = '3d' if input_dim > 2 else 'rectilinear'

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)
    latent1_ax = fig.add_subplot(gs[0:2, 0], aspect='equal')
    latent2_ax = fig.add_subplot(gs[0:2, 1], aspect='equal') #0:2図の大きさ
    error_ax = fig.add_subplot(gs[2, :])
    num_epoch = len(Y_history) #マップの配置

    if input_dim == 3 and latent_dim1 == 2 and latent_dim2 == 2:
        resolution = int(np.sqrt(Y_history.shape[1]))
        if Y_history.shape[1] == resolution ** 2:
            Y_history = np.array(Y_history).reshape((num_epoch, resolution, resolution, input_dim))

    observable_drawer = [None, None, draw_observable_2D,
                         draw_observable_3D][input_dim]
    latent1_drawer = [None, draw_latent_1D, draw_latent_2D][latent_dim1]
    latent2_drawer = [None, draw_latent_1D, draw_latent_2D][latent_dim2]

    ani = FuncAnimation(
        fig,
        update_graph,
        frames=num_epoch,  # // STEP,
        repeat=True,
        interval=10,
        fargs=(latent1_drawer, latent2_drawer, X, Y_history, U_history, V_history, error_history, animal_label, feature_label, fig, latent1_ax, latent2_ax, error_ax, num_epoch))
    plt.show()
    if save_gif:
        ani.save(f"{filename}.mp4", writer='ffmpeg')


def update_graph(epoch, latent1_drawer, latent2_drawer, X, Y_history,
                 U_history, V_history, error_history, animal_label, feature_label,fig, latent1_ax, latent2_ax, error_ax, num_epoch):
    fig.suptitle(f"epoch: {epoch}")
    #  input_ax.view_init(azim=(epoch * 400 / num_epoch), elev=30)
    latent1_ax.cla()
    latent2_ax.cla()
    error_ax.cla()

    Y, U, V= Y_history[epoch], U_history[epoch], V_history[epoch]
    colormap1 = X[:, 0]
    colormap2 = X[0, :]

    latent1_drawer(latent1_ax,U,animal_label,colormap1)
    latent2_drawer(latent2_ax,V,feature_label,colormap2)
    draw_error(error_ax, error_history, epoch)


def draw_observable_3D(ax, X, Y, colormap):
    ax.scatter(X[:, :, 0], X[:, :, 1], X[:, :, 2], c=colormap)
    # ax.set_zlim(-1, 1)
    if len(Y.shape) == 3:
        ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')
        # ax.scatter(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')
    else:
        ax.plot(Y[:,:,0], Y[:,:,1], Y[:,:,2], color='black')
# ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='black')
# ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')


def draw_observable_2D(ax, X, Y, colormap):
    ax.scatter(X[:, 0], X[:, 1], c=colormap)
    ax.plot(Y[:, 0], Y[:, 1], c='black')


def draw_latent_2D(ax, Z, label,colormap):
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    for n in range (Z.shape[0]):
        ax.text(Z[n, 0], Z[n, 1], label[n])
    ax.scatter(Z[:, 0], Z[:, 1], c=colormap)


def draw_latent_1D(ax, Z, colormap):
    ax.scatter(Z, np.zeros(Z.shape), c=colormap)
    ax.set_ylim(-1, 1)

def draw_error(ax, error_history, epoch):
    ax.set_title("error_function", fontsize=8)
    ax.plot(error_history, label='誤差関数')
    ax.scatter(epoch, error_history[epoch], s=55, marker="*")
