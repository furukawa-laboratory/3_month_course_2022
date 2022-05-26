import numpy as np
import matplotlib.pyplot as plt

def load_kura_tsom(xsamples, ysamples, missing_rate=None,retz=None):
    z1 = np.linspace(-1, 1, xsamples)
    z2 = np.linspace(-1, 1, ysamples)

    #z1 = np.random.uniform(-1, 1, xsamples)
    #z2 = np.random.uniform(-1, 1, ysamples)

    z1_repeated, z2_repeated = np.meshgrid(z1, z2, indexing='ij')
    x1 = z1_repeated
    x2 = z2_repeated
    x3 = 1 * (x1**2 - x2**2) + + np.random.uniform(-0.5, 0.5, xsamples*ysamples).reshape(xsamples, ysamples)
    #ノイズを加えたい時はここをいじる,locがガウス分布の平均、scaleが分散,size何個ノイズを作るか
    #このノイズを加えることによって三次元空間のデータ点は上下に動く

    x = np.concatenate([x1[:, :, None], x2[:, :, None], x3[:, :, None]], axis=2)
    truez = np.concatenate([x1[:, None], x2[:, None]], axis=2)

    if missing_rate == 0 or missing_rate == None:
        if retz:
            return x, truez
        else:
            return x

def load_kura_list(xsamples, ysamples, missing_rate=None, retz=None):
    z1 = np.random.uniform(-1, 1, xsamples)
    z2 = np.random.uniform(-1, 1, ysamples)
    z1_num = np.arange(xsamples)
    z2_num = np.arange(ysamples)

    z1_repeated, z2_repeated = np.meshgrid(z1, z2)
    z1_num_repeated, z2_num_repeated = np.meshgrid(z1_num, z2_num)
    x1 = z1_repeated.reshape(-1)
    x2 = z2_repeated.reshape(-1)
    x1_num = z1_num_repeated.reshape(-1)
    x2_num = z2_num_repeated.reshape(-1)
    x3 = x1**2 - x2**2 + np.random.uniform(-0.5, 0.5, xsamples*ysamples)
    #ノイズを加えたい時はここをいじる,locがガウス分布の平均、scaleが分散,size何個ノイズを作るか
    #このノイズを加えることによって三次元空間のデータ点は上下に動く

    x = np.concatenate([x1[:, None], x2[:, None], x3[:, None]], axis=1)
    x_num = np.concatenate([x1_num[:, None], x2_num[:, None]], axis=1)

    if missing_rate == 0 or missing_rate == None:
        if retz:
            return x
        else:
            return x, x_num

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xsamples = 10
    ysamples = 15

    #x = load_kura_tsom(xsamples, ysamples)
    x = load_kura_list(xsamples, ysamples)
    #print(plt.rcParams['image.cmap'])
    fig = plt.figure(figsize=[5, 5])
    ax_x = fig.add_subplot(projection='3d')
    ax_x.scatter(x[:, 0], x[:, 1], x[:, 2])
    ax_x.set_title('Generated three-dimensional data')
    plt.show()