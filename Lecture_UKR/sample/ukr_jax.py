import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
from tqdm import tqdm
config.update("jax_enable_x64", True)

class UKR:
    def __init__(self, X, latent_dim, sigma, prior='random', Zinit=None):
        if X.ndim == 1:
            self.X = X[:, None]
            self.nb_samples = X.shape[0]
            self.ob_dim = 1
        else:
            self.X = X
            self.nb_samples, self.ob_dim = X.shape

        self.sigma = sigma
        self.latent_dim = latent_dim

        if Zinit is None:
            if prior == 'random':
                self.Z = 0.2 * self.sigma * np.random.rand(self.nb_samples, self.latent_dim) - 0.1 * self.sigma
            else:
                self.Z = np.random.normal(0, 0.1 * self.sigma, size=(self.nb_samples, self.latent_dim))
        else:
            self.Z = Zinit
        self.history = {}

    def f(self, Z1, Z2):
        Dzz = Z1[:, None, :] - Z2[None, :, :]
        D = jnp.sum(jnp.square(Dzz), axis=2)
        H = jnp.exp(-0.5 * D / (self.sigma ** 2))
        G = jnp.sum(H, axis=1, keepdims=True)
        R = H / G
        f = R @ self.X
        return f

    def E(self, Z: np.ndarray, X: np.ndarray, alpha=0, norm=2) -> jax.xla.DeviceArray:
        Y = self.f(Z, Z)
        result = (1 / (2 * self.nb_samples)) * jnp.sum((Y - X) ** 2)
        result += (alpha / self.nb_samples) * jnp.sum(Z ** norm)
        return result

    def fit(self, nb_epoch: int, eta: float) -> np.ndarray:
        # 学習過程記録用
        self.history['z'] = np.zeros((nb_epoch, self.nb_samples, self.latent_dim))
        self.history['f'] = np.zeros((nb_epoch, self.nb_samples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
            grad = jax.grad(self.E, argnums=0)(self.Z, self.X)
            self.Z = self.Z - (eta / self.nb_samples) * grad

            # 学習過程記録用
            self.history['z'][epoch] = self.Z
            self.history['f'][epoch] = np.array(self.f(self.Z, self.Z))
            self.history['error'][epoch] = np.array(self.E(self.Z, self.X))

    def calc_approximate_f(self, resolution):
        nb_epoch = self.history['z'].shape[0]
        self.history['y'] = np.zeros((nb_epoch, resolution ** self.latent_dim, self.ob_dim))
        for epoch in tqdm(range(nb_epoch)):
            Z = self.history['z'][epoch, :, :]
            zeta = create_zeta(Z, resolution)
            Y = self.f(zeta, Z)
            self.history['y'][epoch] = Y
        return self.history['y']


def create_zeta(Z, resolution):
    latent_dim = Z.shape[-1]
    mesh, step = np.linspace(Z.min(), Z.max(), resolution, endpoint=False, retstep=True)
    mesh += step / 2.0
    zeta = np.empty((resolution, latent_dim))
    if latent_dim == 1:
        zeta = mesh[:, None]
    elif latent_dim == 2:
        xx, yy = np.meshgrid(mesh, mesh)
        zeta = np.concatenate([xx.reshape(-1)[:, None], yy.reshape(-1)[:, None]], axis=1)
    return zeta


if __name__ == '__main__':
    from Lecture_UKR.data import create_kura
    from Lecture_UKR.data import create_rasen
    from Lecture_UKR.data import create_2d_sin_curve
    from visualizer import visualize_history

    epoch = 200
    sigma = 0.2
    eta = 100
    latent_dim = 2
    resolution = 10
    nb_samples = 125
    seed = 4
    np.random.seed(seed)

    X = create_kura(nb_samples)
    # X = create_rasen(nb_samples)
    # X = create_2d_sin_curve(nb_samples)

    ukr = UKR(X, latent_dim, sigma, prior='random')
    ukr.fit(epoch, eta)
    ukr.calc_approximate_f(resolution)
    visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")



