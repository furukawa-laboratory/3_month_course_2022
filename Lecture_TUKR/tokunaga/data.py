import numpy as np
import matplotlib.pyplot as plt

def load_kura_tsom(xsamples, ysamples, missing_rate=None,retz=True):
    z1 = np.linspace(-1, 1, xsamples)
    z2 = np.linspace(-1, 1, ysamples)

    z1_repeated, z2_repeated = np.meshgrid(z1, z2, indexing='ij')
    x1 = z1_repeated
    x2 = z2_repeated
    x3 = x1**2 - x2**2 + + np.random.uniform(-0.5, 0.5, xsamples*ysamples).reshape(xsamples, ysamples)
    #ノイズを加えたい時はここをいじる,locがガウス分布の平均、scaleが分散,size何個ノイズを作るか
    #このノイズを加えることによって三次元空間のデータ点は上下に動く

    x = np.concatenate([x1[:, :, None], x2[:, :, None], x3[:, :, None]], axis=2)
    truez = np.concatenate([x1[:, :, None], x2[:, :, None]], axis=2)

    if missing_rate == 0 or missing_rate == None:
        if retz:
            return x, truez
        else:
            return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xsamples = 10
    ysamples = 15

    x, truez = load_kura_tsom(xsamples, ysamples)
    print(x.shape)
    fig = plt.figure(figsize=[5, 5])
    ax_x = fig.add_subplot(projection='3d')
    ax_x.scatter(x[:, :, 0], x[:, :, 1], x[:, :, 2])
    ax_x.set_title('Generated three-dimensional data')
    plt.show()