import numpy as np

def create_kura(nb_samples, noise_scale=0.05):
    z1 = np.random.rand(nb_samples) * 2.0 - 1.0 #-1~1まで
    z2 = np.random.rand(nb_samples) * 2.0 - 1.0
    X = np.zeros((nb_samples,3))
    X[:, 0] = z1
    X[:, 1] = z2
    X[:, 2] = 0.5 * (z1 ** 2 - z2 ** 2)
    X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)
    return X

def create_rasen(nb_samples):
    z = np.linspace(-3 * np.pi, 3 * np.pi, nb_samples)
    X = np.zeros((nb_samples, 3))
    X[:, 0] = np.cos(z)
    X[:, 1] = np.sin(z)
    X[:, 2] = z
    return X

def create_2d_sin_curve(nb_samples, noise_scale=0.01):
    z = np.random.rand(nb_samples) * 2.0 - 1.0
    X = np.zeros((nb_samples, 2))
    X[:, 0] = z
    X[:, 1] = np.sin(z * np.pi) * 0.5
    X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)
    return X


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    nb_samples = 200
    X = create_kura(nb_samples, noise_scale=0.05)
    # X = create_rasen(nb_samples)
    # X = create_2d_sin_curve(nb_samples, noise_scale=0.01)

    _, D = X.shape
    projection = '3d' if D >= 3 else 'rectilinear'

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=projection)

    xs = (X[:, 0], X[:, 1], X[:, 2]) if D >= 3 else (X[:, 0], X[:, 1])
    ax.scatter(*xs)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlabel("x3")

    plt.show()


    # 機械学習の課題
    # import random
    # A=0
    # for a in range(1000000):
    #     d = random.randint(1, 6)
    #     if (d == 1) | (d == 2):
    #         A+=1
    # print(str(100*A/1000000)+"%")
    #
    # bag = 2  # 袋の総数
    # bag_x = 1  # 袋Xの数
    # bag_y = 1  # 袋Yの数
    #
    # ball_in_bag_x = [4, 6]  # 袋Xの中身(赤玉が4個、白玉が6個)
    # ball_in_bag_y = [5, 2]  # 袋Yの中身(赤玉が5個、白玉が2個)
    #
    #
    # def jouhou(bag, bag_sum, ball_in_bag):
    #     ans = (ball_in_bag[0] / sum(ball_in_bag)) * (bag / bag_sum)
    #     return ans
    #
    #
    # j = jouhou(bag_x, bag, ball_in_bag_x)
    # print(str(j*100) + "%")