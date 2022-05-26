import numpy as np
import matplotlib.pyplot as plt

def load_kura_tsom(xsamples, ysamples, missing_rate=None,retz=False):
    # z1 = np.linspace(-1, 1, xsamples) #randam関数にする
    # z2 = np.linspace(-1, 1, ysamples)
    # z1 = np.random.normal(0, 0.5, xsamples)
    # z2 = np.random.normal(0, 0.5, ysamples)
    # z1 = np.random.randn(xsamples)
    # z2 = np.random.randn(ysamples)
    z1 = np.random.uniform(-1, 1, xsamples)
    z2 = np.random.uniform(-1, 1, ysamples)


    z1_repeated, z2_repeated = np.meshgrid(z1, z2)
    x1 = z1_repeated
    x2 = z2_repeated
    x3 = x1**2 - x2**2
    #ノイズを加えたい時はここをいじる,locがガウス分布の平均、scaleが分散,size何個ノイズを作るか
    #このノイズを加えることによって三次元空間のデータ点は上下に動く

    x = np.concatenate((x1[:, :, np.newaxis], x2[:, :, np.newaxis], x3[:, :, np.newaxis]), axis=2)
    truez = np.concatenate((z2_repeated[:, :, np.newaxis], z2_repeated[:, :, np.newaxis]), axis=2)

    if missing_rate == 0 or missing_rate == None:
        if retz:
            return x, truez
        else:
            return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xsamples = 50
    ysamples = 30

    x, truez = load_kura_tsom(xsamples, ysamples, retz=True)
    
    fig = plt.figure(figsize=[5, 5])
    ax_x = fig.add_subplot(projection='3d')
    ax_x.scatter(x[:, :, 0], x[:, :, 1], x[:, :, 2])
    ax_x.set_title('Generated three-dimensional data')
    plt.show()


