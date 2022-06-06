import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler
STEP = 150


def visualize_history(X, Y_history, U_history, V_history, error_history, save_gif=False, filename="tmp", zzz=None):
    input_dim, latent_dim1, latent_dim2 = X.shape[2], U_history[0].shape[1], V_history[0].shape[1]
    input_projection_type = '3d' if input_dim > 2 else 'rectilinear'

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 3)
    input_ax = fig.add_subplot(gs[0:2, 0], projection=input_projection_type)
    latent_ax1 = fig.add_subplot(gs[0:2, 1], aspect='equal')
    latent_ax1.set_facecolor('k')
    latent_ax2 = fig.add_subplot(gs[0:2, 2], aspect='equal')
    latent_ax2.set_facecolor('k')
    error_ax = fig.add_subplot(gs[2, :])
    num_epoch = len(Y_history)

    # if input_dim == 3 and latent_dim1 == 2:
    #     resolution = int(np.sqrt(Y_history.shape[1]))
    #     if Y_history.shape[1] == resolution ** 2:
    #         Y_history = np.array(Y_history).reshape((num_epoch, resolution, resolution, input_dim))
    #
    # if input_dim == 3 and latent_dim2 == 2:
    #     resolution = int(np.sqrt(Y_history.shape[1]))
    #     if Y_history.shape[1] == resolution ** 2:
    #         Y_history = np.array(Y_history).reshape((num_epoch, resolution, resolution, input_dim))
    observable_drawer = [None, None, draw_observable_2D,
                         draw_observable_3D][input_dim]

    latent_drawer1 = [None, draw_latent_1D, draw_latent_2D][latent_dim1]
    latent_drawer2 = [None, draw_latent_1D, draw_latent_2D][latent_dim2]

    ani = FuncAnimation(
        fig,
        update_graph,
        frames=num_epoch,  # // STEP,
        repeat=True,
        interval=50,
        fargs=(observable_drawer, latent_drawer1, latent_drawer2, X, Y_history, U_history, V_history, error_history, fig,
               input_ax, latent_ax1, latent_ax2, error_ax, num_epoch, zzz))
    plt.show()
    if save_gif:
        ani.save(f"{filename}.mp4", writer='ffmpeg')
def update_graph(epoch, observable_drawer, latent_drawer1,latent_drawer2, X, Y_history,
                 U_history, V_history, error_history, fig, input_ax, latent_ax1, latent_ax2, error_ax, num_epoch, zzz):
    fig.suptitle(f"epoch: {epoch}")
    input_ax.cla()
    #  input_ax.view_init(azim=(epoch * 400 / num_epoch), elev=30)
    latent_ax1.cla()
    latent_ax2.cla()
    error_ax.cla()

    Y, U, V= Y_history[epoch], U_history[epoch], V_history[epoch]
    (truez, z1, z2) = zzz
    mmscaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    truez = mmscaler.fit_transform(truez.reshape(-1, 2))
    truez = truez.reshape(z1.shape[0], z2.shape[0], 2)

    z1 = mmscaler.fit_transform(z1[:, None])
    z2 = mmscaler.fit_transform(z2[:, None])
    colormap = np.ones((truez.shape[0], truez.shape[1], 3))
    # print(colormap.shape)
    colormap[:, :, 0] = truez[:, :, 0]
    colormap[:, :, 2] = truez[:, :, 1]
    colormap = colormap.reshape(-1, 3)
    colormap1 = np.ones((z1.shape[0], 3))
    colormap1[:, 0] = z1[:, 0]

    colormap2 = np.ones((z2.shape[0], 3))
    colormap2[:, 2] = z2[:, 0]

    # print(V)

    observable_drawer(input_ax, X, Y, colormap)
    latent_drawer1(latent_ax1, U,  colormap1)
    latent_drawer2(latent_ax2, V, colormap2)
    draw_error(error_ax, error_history, epoch)


# def visualize_history(X, Y_history, U_history, V_history, error_history, save_gif=False, filename="tmp", label1 = None, label2 = None):
#     input_dim, latent_dim1, latent_dim2 = X.shape[2], U_history[0].shape[1], V_history[0].shape[1]
#     input_projection_type = '3d' if input_dim > 2 else 'rectilinear'
#
#     fig = plt.figure(figsize=(10, 8))
#     gs = fig.add_gridspec(3, 2)
#     # input_ax = fig.add_subplot(gs[0:2, 0], projection=input_projection_type)
#     latent_ax1 = fig.add_subplot(gs[0:2, 0], aspect='equal')
#     latent_ax2 = fig.add_subplot(gs[0:2, 1], aspect='equal')
#     error_ax = fig.add_subplot(gs[2, :])
#     num_epoch = len(Y_history)
#
#     # if input_dim == 3 and latent_dim1 == 2:
#     #     resolution = int(np.sqrt(Y_history.shape[1]))
#     #     if Y_history.shape[1] == resolution ** 2:
#     #         Y_history = np.array(Y_history).reshape((num_epoch, resolution, resolution, input_dim))
#     #
#     # if input_dim == 3 and latent_dim2 == 2:
#     #     resolution = int(np.sqrt(Y_history.shape[1]))
#     #     if Y_history.shape[1] == resolution ** 2:
#     #         Y_history = np.array(Y_history).reshape((num_epoch, resolution, resolution, input_dim))
#     observable_drawer = [None, None, draw_observable_2D,
#                          draw_observable_3D][input_dim]
#
#     latent_drawer1 = [None, draw_latent_1D, draw_latent_2D][latent_dim1]
#     latent_drawer2 = [None, draw_latent_1D, draw_latent_2D][latent_dim2]
#
#     ani = FuncAnimation(
#         fig,
#         update_graph,
#         frames=num_epoch,  # // STEP,
#         repeat=True,
#         interval=50,
#         fargs=(observable_drawer, latent_drawer1, latent_drawer2, X, Y_history, U_history, V_history, error_history, fig,
#                latent_ax1, latent_ax2, error_ax, num_epoch, label1, label2))
#     plt.show()
#     if save_gif:
#         ani.save(f"{filename}.mp4", writer='ffmpeg')
#
# def update_graph(epoch, observable_drawer, latent_drawer1,latent_drawer2, X, Y_history,
#                  U_history, V_history, error_history, fig, latent_ax1, latent_ax2, error_ax, num_epoch, label1, label2):
#     fig.suptitle(f"epoch: {epoch}")
#     # input_ax.cla()
#     #  input_ax.view_init(azim=(epoch * 400 / num_epoch), elev=30)
#     latent_ax1.cla()
#     latent_ax2.cla()
#     error_ax.cla()
#
#     Y, U, V= Y_history[epoch], U_history[epoch], V_history[epoch]
#     colormap = X[:, :, 0]
#     colormap1 = U[:, 0]
#     colormap2 = V[:, 0]
#     # print(V)
#
#     observable_drawer(input_ax, X, Y, colormap)
#     latent_drawer1(latent_ax1, U,  colormap1, label1)
#     latent_drawer2(latent_ax2, V, colormap2, label2)
#     draw_error(error_ax, error_history, epoch)


def draw_observable_3D(ax, X, Y, colormap):
    # print(X.shape,type(colormap))
    # print(colormap.shape)
    ax.scatter(X[:, :, 0], X[:, :, 1], X[:, :, 2], c=colormap)
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
    # for i in range (Z.shape[0]):
    #      ax.annotate(label[i], xy = (Z[i, 0], Z[i, 1]))

def draw_latent_1D(ax, Z, colormap):
    ax.scatter(Z, np.zeros(Z.shape), c=colormap, )
    ax.set_ylim(-1, 1)
    # for i in range (Z.shape[0]):
    #      ax.annotate(label[i], xy = (Z[i, 0], np.zeros(1)))

def draw_error(ax, error_history, epoch):
    ax.set_title("error_function", fontsize=8)
    ax.plot(error_history, label='誤差関数')
    ax.scatter(epoch, error_history[epoch], s=55, marker="*")
