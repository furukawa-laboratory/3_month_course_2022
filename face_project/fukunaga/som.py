import numpy as np
from tqdm import tqdm #プログレスバーを表示させてくれる
from matplotlib import pyplot as plt

class SOM:
    def __init__(self, X, latent_dim, sigma, tau, sigmamin, K):
        self.X = X
        self.input_dim = X.shape[1]
        self.sigma =sigma
        self.latent_dim =latent_dim
        self.K = K
        self.tau =tau
        self.sigmamin = sigmamin
        self.zeta = create_Zeta2D()
        self.Y = np.random.uniform(-1, 1, X.shape[0]*self.input_dim).reshape(X.shape[0], self.input_dim)
        # print(self.X.shape)
        # print(self.Y.shape)



    def argmin_Z(self):

        Dist = np.sum((self.Y[:, None, :] - self.X[None, :, :])**2, axis=2)
        # print(Dist.shape)
        # print(self.Y)
        K_star = np.argmin(Dist, axis=0)
        # print(K_star.shape)
        self.Z = self.zeta[K_star]
        # print(self.Z.shape)






    def f(self):
        d = np.sum((self.zeta[:, None, :] - self.Z[None, :, :]) ** 2, axis=2)
        H = -1*(d/(2*self.sigma**2))
        h = np.exp(H)
        bunshi = h@self.X
        bunbo = np.sum(h, axis=1, keepdims=True)
        print(bunbo)
        self.Y = bunshi/bunbo
        # print(222222222222222)
        # print(self.Y)



    def update_sigma(self,epoch):
        sig = max(self.sigma *(1-(epoch/self.tau)), self.sigmamin)

        self.sigma =sig
        # print(self.sigma)





    def fit(self, nb_epoch, plt_is=True):

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(1, 2)
        input_ax = fig.add_subplot(gs[0:2, 0], projection='3d')
        latent_ax = fig.add_subplot(gs[0:2, 1])

        for epoch in range(nb_epoch):
            self.argmin_Z()
            self.f()
            self.update_sigma(epoch)

            if (plt_is == True):# and epoch % 10 ==0 ):
                draw(input_ax, latent_ax, self.X, self.Y, self.Z)

def draw(input_ax, latent_ax, X, Y, Z ):
    input_ax.cla()
    latent_ax.cla()
    input_ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0])
    input_ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='k')
    latent_ax.scatter(Z[:, 0], Z[:, 1], c=X[:, 0])
    plt.pause(1)


def creat_Zeta1D():
    A = np.linspace(-1, 1, K)
    return A

def create_Zeta2D():
    A = np.linspace(-1, 1, K)
    B = np.linspace(-1, 1, K)
    XX, YY = np.meshgrid(A, B)
    xx = XX.reshape(-1)
    yy = YY.reshape(-1)
    zeta = np.concatenate([xx[:, None], yy[:, None]], axis=1)

    return zeta



if __name__ == '__main__':
    nb_samples = 100
    K = 10

    sigmamin = 0.1
    epoch = 300  # 学習回数
    tau = 150
    sigma = 2  # カーネルの幅
    latent_dim = 2  # 潜在空間の次元
    seed = 4
    np.random.seed(seed)




    def create_kura(nb_samples, noise_scale=0.05):
        z1 = np.random.rand(nb_samples) * 2.0 - 1.0  # -1~1まで
        z2 = np.random.rand(nb_samples) * 2.0 - 1.0
        X = np.zeros((nb_samples, 3))
        X[:, 0] = z1
        X[:, 1] = z2
        X[:, 2] = 0.5 * (z1 ** 2 - z2 ** 2)
        X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)
        return X


    X = create_kura(nb_samples)
    som = SOM(X, latent_dim, sigma, tau, sigmamin, K)
    som.fit(epoch)


















