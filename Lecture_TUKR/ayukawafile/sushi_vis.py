import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

STEP = 150

def visualize_history(X, Y_history, Z_history, v_history, error_history, datalabel,
                      save_gif=True,
                      filename="tmp"):
    input_dim, xlatent_dim = X.shape[2], Z_history[0].shape[1]
    input_projection_type = '3d' if input_dim > 2 else 'rectilinear'


    ylatent_dim = v_history[0].shape[1]
    # yinput_projection_type = '3d' if yinput_dim > 2 else 'rectilinear'

    # print(X.shape)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)


    xlatent_ax = fig.add_subplot(gs[0:2, 0], aspect='equal')
    ylatent_ax = fig.add_subplot(gs[0:2, 1], aspect='equal')
    error_ax = fig.add_subplot(gs[2, :])
    num_epoch = len(Y_history)

    if input_dim == 3 and xlatent_dim == 2 and ylatent_dim == 2:
        resolution = int(np.sqrt(Y_history.shape[1]))


    observable_drawer = [None, None, draw_observable_2D,
                         draw_observable_3D][input_dim]
    xlatent_drawer = [None, draw_latent_1D, draw_latent_2DX][xlatent_dim]
    ylatent_drawer = [None, draw_latent_1D, draw_latent_2DY][ylatent_dim]

    ani = FuncAnimation(
        fig,
        update_graph,
        frames=num_epoch,  # // STEP,
        repeat=True,
        interval=50,
        fargs=(observable_drawer, xlatent_drawer, ylatent_drawer, X, Y_history, Z_history, v_history, error_history, fig, xlatent_ax, ylatent_ax, error_ax, num_epoch, datalabel))
    plt.show()


def visualize_fig_history(X, Y_history, Z_history, v_history, error_history, datalabel, save_gif=True, filename="tmp"):
    input_dim, xlatent_dim = X.shape[2], Z_history[0].shape[1]
    input_projection_type = '3d' if input_dim > 2 else 'rectilinear'

    # print(input_dim)
    # input_ddim=input_dim[0]
    ylatent_dim = v_history[0].shape[1]

    # print(input_ddim)
    # exit()
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)

    xlatent_ax = fig.add_subplot(gs[0:2, 0], aspect='equal')
    ylatent_ax = fig.add_subplot(gs[0:2, 1], aspect='equal')
    error_ax = fig.add_subplot(gs[2, :])
    num_epoch = len(Y_history)

    if input_dim == 3 and xlatent_dim == 2 and ylatent_dim == 2:
        resolution = int(np.sqrt(Y_history.shape[1]))

    xlatent_drawer = [None, draw_latent_1D, draw_latent_2DX][xlatent_dim]
    ylatent_drawer = [None, draw_latent_1D, draw_latent_2DY][ylatent_dim]



    xlatent_drawer(xlatent_ax, Z_history[-1], 1)
    ylatent_drawer(ylatent_ax, v_history[-1], 1, datalabel[1])


    plt.show()
    if save_gif:
        fig.savefig(f"{filename}.png")







def update_graph(epoch, observable_drawer, xlatent_drawer, ylatent_drawer, X, Y_history,
                 Z_history, v_history, error_history, fig,
                 # input_ax,
                 xlatent_ax, ylatent_ax, error_ax, num_epoch, datalabel):
    fig.suptitle(f"epoch: {epoch}")
    # input_ax.cla()
    # print(datalabel)
    # exit()
    # input_ax.view_init(azim=(epoch * 400 / num_epoch), elev=30)
    xlatent_ax.cla()
    ylatent_ax.cla()
    error_ax.cla()

    Y, Z, v = Y_history[epoch], Z_history[epoch], v_history[epoch]
    colormap = X[:, :, 0]
    # print(datalabel[1])
    xlatent_drawer(xlatent_ax, Z, X[:,0,0])
    # ylatent_drawer(ylatent_ax, v, X[0,:,1]) #鞍型データ用
    ylatent_drawer(ylatent_ax, v, X[0,:,0], datalabel[1])
    draw_error(error_ax, error_history, epoch)


def draw_observable_3D(ax, X, Y, Z, colormap):
    Z=np.tile(X[:, :, 0],(1))#zを縦に２０回繰り返そう
    ax.scatter(X[:, :, 0], X[:, :, 1], X[:, :, 2], c=X[:, :, 0])
    if len(Y.shape) == 3:
        ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')
    else:
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='black')



def draw_observable_2D(ax, X, Y, colormap):
    ax.scatter(X[:, 0], X[:, 1], c='blue')
    ax.plot(Y[:, 0], Y[:, 1], c='black')


def draw_latent_2DX(ax, Z, c):
    roop=Z.shape[0]
    # print(Z.shape[0])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.scatter(Z[:, 0], Z[:, 1], c='red', s=1)
    # print(label)
    # print(roop)
    # for i in range(roop):
    #     ax.annotate(datalabel[i], xy=(Z[i, 0], Z[i, 1]), size=10, color="black")

def draw_latent_2DY(ax, Z, c, datalabel):
    roop=Z.shape[0]
    # print(Z.shape[0])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.scatter(Z[:, 0], Z[:, 1], c='blue', s=1)
    # print(label)
    # print(roop)
    for i in range(roop):
        ax.annotate(datalabel[i], xy=(Z[i, 0], Z[i, 1]), size=10, color="black")


def draw_latent_1D(ax, Z, X):
    # ax.scatter(Z, np.zeros(Z.shape), c='blueviolet')

    ax.scatter(Z, np.zeros(Z.shape), c=X)
    ax.set_ylim(-1, 1)
    # roop=Z.shape[0]
    # print(roop)
    # for i in range(roop):
    #     ax.annotate(datalabel[i], xy=(Z[i, 0], Z[i, 1]), size=10, color="black")

def draw_error(ax, error_history, epoch):
    ax.set_title("error_function", fontsize=8)
    ax.plot(error_history, label='誤差関数')
    ax.scatter(epoch, error_history[epoch], s=55, marker="*")
