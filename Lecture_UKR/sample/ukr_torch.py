import numpy as np
import torch
from tqdm import tqdm
torch.set_default_tensor_type(torch.DoubleTensor)

class UKR:
    def __init__(self, X, latent_dim, sigma, prior='random', Zinit=None):
        if X.ndim == 1:
            self.X = torch.from_numpy(X[:, None])
            self.nb_samples = X.shape[0]
            self.ob_dim = 1
        else:
            self.X = torch.from_numpy(X)
            self.nb_samples, self.ob_dim = X.shape

        self.sigma = sigma
        self.latent_dim = latent_dim

        if Zinit is None:
            if prior == 'random':
                Z = 0.2 * self.sigma * np.random.rand(self.nb_samples, self.latent_dim) - 0.1 * self.sigma
            else:
                Z = np.random.normal(0, 0.1 * self.sigma, size=(self.nb_samples, self.latent_dim))
        else:
            Z = Zinit
        self.Z = torch.tensor(Z, requires_grad=True)

        self.history = {}

    def f(self, Z1: torch.Tensor, Z2: torch.Tensor):
        D = torch.sum((Z1[:, None, :] - Z2[None, :, :]) ** 2, dim=2)
        H = torch.exp(-0.5 * D / (self.sigma ** 2))
        G = torch.sum(H, dim=1, keepdim=True)
        R = H / G
        f = R @ self.X
        return f

    def E(self, Z: torch.Tensor, X: torch.Tensor, alpha=0, norm=2) -> torch.Tensor:
        Y = self.f(Z, Z)
        result = (1 / (2 * self.nb_samples)) * torch.sum((Y - X) ** 2)
        result += (alpha / self.nb_samples) * torch.sum(Z ** norm)
        return result

    def fit(self, nb_epoch: int, eta: float) -> np.ndarray:
        # 学習過程記録用
        self.history['z'] = np.zeros((nb_epoch, self.nb_samples, self.latent_dim))
        self.history['f'] = np.zeros((nb_epoch, self.nb_samples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
            self.E(self.Z, self.X).backward()
            with torch.no_grad():
                self.Z = self.Z - (eta / self.nb_samples) * self.Z.grad
            self.Z.requires_grad = True

            # 学習過程記録用
            self.history['z'][epoch] = self.Z.detach().numpy()
            self.history['f'][epoch] = self.f(self.Z, self.Z).detach().numpy()
            self.history['error'][epoch] = self.E(self.Z, self.X).detach().numpy()

    def calc_approximate_f(self, resolution):
        nb_epoch = self.history['z'].shape[0]
        self.history['y'] = np.zeros((nb_epoch, resolution ** self.latent_dim, self.ob_dim))
        for epoch in range(nb_epoch):
            Z = self.history['z'][epoch, :, :]
            zeta = create_zeta(Z, resolution)
            Y = self.f(zeta, Z)
            self.history['y'][epoch] = Y.detach().numpy()
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
    return torch.tensor(zeta)


if __name__ == '__main__':
    from Lecture_UKR.data import create_kura
    from Lecture_UKR.data import create_rasen
    from Lecture_UKR.data import create_2d_sin_curve
    from visualizer import visualize_history

    epoch = 200
    sigma = 0.2
    eta = 100
    latent_dim = 2
    resolution = 15
    nb_samples = 100
    seed = 4
    np.random.seed(seed)

    X = create_kura(nb_samples)
    # X = create_rasen(nb_samples)
    # X = create_2d_sin_curve(nb_samples)

    ukr = UKR(X, latent_dim, sigma, prior='random')
    ukr.fit(epoch, eta)
    ukr.calc_approximate_f(resolution)
    visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")



