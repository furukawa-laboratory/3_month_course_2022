import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

STEP = 150
# (X_TUKR[0][:, :, None], Tukr.history['f'], Tukr.history['z'], Tukr.history['v'], Tukr.history['error'], X_TUKR, save_gif=True, filename="tmp")
#                       X, Y_history, Z_history, v_history, error_history, datalabel, save_gif=True, filename="tmp"
def visualize_history(X, Y_history, Z_history, v_history, error_history, datalabel,
                      save_gif=True,
                      filename="tmp"):
    input_dim, xlatent_dim = X.shape[2], Z_history[0].shape[1]
    input_projection_type = '3d' if input_dim > 2 else 'rectilinear'

    # print(input_dim)
    # exit()

    ylatent_dim = v_history[0].shape[1]
    # yinput_projection_type = '3d' if yinput_dim > 2 else 'rectilinear'

    # print(X.shape)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)

    # input_ax = fig.add_subplot(gs[1:2, 1], projection=input_projection_type)


    # yinput_ax = fig.add_subplot(gs[0:2, 0], projection=yinput_projection_type)
    xlatent_ax = fig.add_subplot(gs[0:2, 0], aspect='equal')
    ylatent_ax = fig.add_subplot(gs[0:2, 1], aspect='equal')
    error_ax = fig.add_subplot(gs[2, :])
    num_epoch = len(Y_history)

    # fig = plt.figure(figsize=(10, 8))
    # gs = fig.add_gridspec(3, 2)
    # input_ax = fig.add_subplot(gs[0:2, 0], projection=input_projection_type)
    # latent_ax = fig.add_subplot(gs[0:2, 1], aspect='equal')
    # error_ax = fig.add_subplot(gs[2, :])
    # num_epoch = len(Y_history)
    # print((num_epoch, int(np.sqrt(Y_history.shape[1])), int(np.sqrt(Y_history.shape[1])), input_dim))
    if input_dim == 3 and xlatent_dim == 2 and ylatent_dim == 2:
        resolution = int(np.sqrt(Y_history.shape[1]))
        # if Y_history.shape[1] == resolution ** 2:
            # Y_history = np.array(Y_history).reshape((num_epoch, resolution, resolution, input_dim))

    # if yinput_dim == 3 and ylatent_dim == 2:
    #     resolution = int(np.sqrt(Y_history.shape[1]))
    #     if Y_history.shape[1] == resolution ** 2:
    #         Y_history = np.array(Y_history).reshape((num_epoch, resolution, resolution, yinput_dim))

    observable_drawer = [None, None, draw_observable_2D,
                         draw_observable_3D][input_dim]
    xlatent_drawer = [None, draw_latent_1D, draw_latent_2D][xlatent_dim]
    ylatent_drawer = [None, draw_latent_1D, draw_latent_2D][ylatent_dim]

    ani = FuncAnimation(
        fig,
        update_graph,
        frames=num_epoch,  # // STEP,
        repeat=True,
        interval=50,
        # fargs=(observable_drawer, xlatent_drawer, ylatent_drawer, X, Y_history, Z_history, v_history, error_history, fig, input_ax, xlatent_ax, ylatent_ax, error_ax, num_epoch, datalabel))
        fargs = (observable_drawer, xlatent_drawer, ylatent_drawer, X, Y_history, Z_history, v_history, error_history, fig, xlatent_ax, ylatent_ax, error_ax, num_epoch, datalabel))
    plt.show()
    # if save_gif:
    #     ani.save(f"{filename}.mp4", writer='ffmpeg')

# X[0][:, :, None], ukr.history['x'], ukr.history['z'], ukr.history['v'], ukr.history['error'], X, save_gif=True, filename="tmp")
# X[0][:, :, None], ukr.history['x'], ukr.history['z'], ukr.history['v'], ukr.history['error'], X, save_gif=True, filename="tmp"
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

    # observable_drawer = [None, None, draw_observable_2D, draw_observable_3D][input_dim]
    # observable_drawer = [None, draw_observable_2D, draw_observable_3D][input_dim]
    xlatent_drawer = [None, draw_latent_1D, draw_latent_2D][xlatent_dim]
    ylatent_drawer = [None, draw_latent_1D, draw_latent_2D][ylatent_dim]

    # print(datalabel)
    # print("--------------------------------")
    # print(X.shape)
    # print("--------------------------------")
    # print(Y_history[-1].shape)
    # print("--------------------------------")
    # print(Y_history.shape)


    # observable_drawer(input_dim,X,Y_history[-1], 'blue')
    xlatent_drawer(xlatent_ax, Z_history[-1], 1, datalabel[1])
    ylatent_drawer(ylatent_ax, v_history[-1], 1, datalabel[2])
    # error_drawer(error_ax, error_history[-1], 'b')


    plt.show()
    if save_gif:
        fig.savefig(f"{filename}.png")







def update_graph(epoch, observable_drawer, xlatent_drawer, ylatent_drawer, X, Y_history,
                 Z_history, v_history, error_history, fig,
                 # input_ax,
                 xlatent_ax, ylatent_ax, error_ax, num_epoch, datalabel):
    fig.suptitle(f"epoch: {epoch}")
    # input_ax.cla()

    #  input_ax.view_init(azim=(epoch * 400 / num_epoch), elev=30)
    xlatent_ax.cla()
    ylatent_ax.cla()
    error_ax.cla()

    Y, Z, v = Y_history[epoch], Z_history[epoch], v_history[epoch]
    colormap = X[:, :, 0]
    # print(input_ax)
    # print('----------------------------------')
    # print(X.shape)
    # print('----------------------------------')
    # print(Y.shape)
    # print('----------------------------------')
    # print(v.shape)
    # print('----------------------------------')
    # print(colormap.shape)
    # print('----------------------------------')
    #observable_drawer(input_ax, X, Y, v, colormap)
    xlatent_drawer(xlatent_ax, Z, X[:,0,0], datalabel[1])
    # ylatent_drawer(ylatent_ax, v, X[0,:,1]) #鞍型データ用
    ylatent_drawer(ylatent_ax, v, X[0,:,0], datalabel[2])
    draw_error(error_ax, error_history, epoch)


def draw_observable_3D(ax, X, Y, Z, colormap):
    Z=np.tile(X[:, :, 0],(1))#zを縦に２０回繰り返そう
    ax.scatter(X[:, :, 0], X[:, :, 1], X[:, :, 2], c=X[:, :, 0])
    # ax.set_zlim(-1, 1)
    if len(Y.shape) == 3:
        ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')
        # ax.scatter(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')
    else:
        ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='black')
# ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='black')
# ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')


def draw_observable_2D(ax, X, Y, colormap):
    ax.scatter(X[:, 0], X[:, 1], c='blue')
    ax.plot(Y[:, 0], Y[:, 1], c='black')


def draw_latent_2D(ax, Z, c, datalabel):
    roop=Z.shape[0]
    # print(Z.shape[0])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.scatter(Z[:, 0], Z[:, 1], c='blue', s=10)
    # print(label)
    # print(roop)
    for i in range(roop):
        ax.annotate(datalabel[i], xy=(Z[i, 0], Z[i, 1]), size=10, color="black")




    #ax.annotate("animal", xy = (Z[:, 0], Z[:, 1]), size = 15, color = "red")

def draw_latent_1D(ax, Z, X):
    # ax.scatter(Z, np.zeros(Z.shape), c='blueviolet')

    ax.scatter(Z, np.zeros(Z.shape), c=X)
    ax.set_ylim(-1, 1)

def draw_error(ax, error_history, epoch):
    ax.set_title("error_function", fontsize=8)
    ax.plot(error_history, label='誤差関数')
    ax.scatter(epoch, error_history[epoch], s=55, marker="*")





