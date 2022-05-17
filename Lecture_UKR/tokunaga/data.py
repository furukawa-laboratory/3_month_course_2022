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
    X[:, 1] = np.sin(z * np.pi)
    X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)
    return X


def create_big_kura(nb_samples, noise_scale=0.05):
    z1 = np.random.rand(nb_samples) * 2.0 - 1.0 #-1~1まで
    z2 = np.random.rand(nb_samples) * 2.0 - 1.0
    X = np.zeros((nb_samples,3))
    X[:, 0] = z1
    X[:, 1] = z2
    X[:, 2] = np.sin(1.2 * z1 * np.pi)
    X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)
    return X

def create_cluster(nb_samples, noise_scale=0.05):
    cnb_samples = int(nb_samples/4)
    n = 0.1
    m = 1
    zc11 = np.random.uniform(-m, -n, cnb_samples)
    zc12 = np.random.uniform(-m, -n, cnb_samples)
    zc21 = np.random.uniform(n, m, cnb_samples)
    zc22 = np.random.uniform(n, m, cnb_samples)
    zc31 = np.random.uniform(-m, -n, cnb_samples)
    zc32 = np.random.uniform(n, m, cnb_samples)
    zc41 = np.random.uniform(n, m, cnb_samples)
    zc42 = np.random.uniform(-m, -n, cnb_samples)
    X = np.zeros((nb_samples, 3))
    X[:, 0] = np.concatenate((zc11, zc21, zc31, zc41), axis=0)
    X[:, 1] = np.concatenate((zc12, zc22, zc32, zc42), axis=0)
    X[:, 2] = 0.5 * (X[:, 0]**2 - X[:, 1]**2)
    return X
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    nb_samples = 200
    #X = create_kura(nb_samples, noise_scale=0.05)
    # X = create_rasen(nb_samples)
    # X = create_2d_sin_curve(nb_samples, noise_scale=0.01)
    #X = create_big_kura(nb_samples, noise_scale=0.05)
    X = create_cluster(nb_samples)

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