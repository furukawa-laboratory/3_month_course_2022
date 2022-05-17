import numpy as np
from tqdm import tqdm #プログ
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

class UKR:
    def __init__(self, X, latent_dim, sigma, prior='random', Zinit=None):
        #--------初期値を設定する．---------
        self.X = X
        #ここから下は書き換えてね
        self.nb_samples, self.ob_dim = X.shape
        self.sigma = sigma
        self.latent_dim = latent_dim

        if Zinit is None:
            if prior == 'random': #一様事前分布のとき
                Z = np.random.uniform(-0.001, 0.001, self.nb_samples*latent_dim).reshape(self.nb_samples,self.latent_dim)
            else: #ガウス事前分布のとき
                Z = np.random.normal(0, 0.001, self.nb_samples*latent_dim).reshape(self.nb_samples, self.latent_dim)
        else: #Zの初期値が与えられた時
            Z = Zinit
        self.Z = Z
        self.history = {}

    def f(self, Z1, Z2): #写像の計算
        s = jnp.sum((Z1[:,None,:]-Z2[None,:,:])**2, axis=2)
        k = jnp.exp(-1*s/(2*self.sigma**2))
        f = k@self.X/jnp.sum(k, axis=1, keepdims=True)

        return f

    def forf(self, Z1, Z2):
        s = np.zeros((Z1.shape[0], Z2.shape[0]))
        for i in range(Z1.shape[0]):
            for j in range(Z2.shape[0]):
                for k in range(self.latent_dim):
                    s[i, j] += (Z1[i, k] - Z2[j, k]) ** 2

        k = np.exp(-1 * s / (2 * self.sigma ** 2))

        kxsum = np.zeros((k.shape[0], self.X.shape[1]))
        ksum = np.zeros(k.shape[0])
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                ksum[i] += k[i, j]
                for d in range(self.X.shape[1]):
                    kxsum[i, d] += k[i, j] * self.X[j, d]

        f = np.zeros(kxsum.shape)

        for i in range(self.nb_samples):
            for k in range(self.ob_dim):
                f[i, k] = kxsum[i, k] / ksum[i]

        return f

    def E(self, Z, X, alpha=0.0001, norm=2): #目的関数の計算
        d = ((X-self.f(Z, Z))**2)/self.nb_samples
        E = jnp.sum(d)+alpha*jnp.sum(Z**norm)
        return E

    def SE(self, Z, X, alpha=0.001, norm=2):
        d = 0
        R = 0
        f = self.f(Z, Z)
        for i in range(self.nb_samples):
            for od in range(self.ob_dim):
                d += (X[i, od] - f[i, od])**2
            for ld in range(latent_dim):
                R += Z[i, ld]**2

        #print("d=" + str(d/self.nb_samples) + "  R=" + str(alpha*R**norm/self.nb_samples))
        return (d+alpha*R**norm)/self.nb_samples

    # def dEdZ(self, Z, X):
    #     r = np.zeros((self.nb_samples, self.nb_samples))
    #     d = np.zeros((self.nb_samples, self.nb_samples))
    #     delta = np.zeros((self.nb_samples, self.nb_samples))
    #     f = self.f(Z, Z)
    #     for i in range(self.nb_samples):
    #         for j in range(self.nb_samples):
    #             for l in range(self.ob_dim):
    #                 d[i, j] += self.d(X[i, l], f[j, l])
    #         r[i, :] += self.k(Z[i, :], Z)/self.K(Z[i, :], Z)
    #
    #     dT = d.T
    #
    #     for i in range(self.nb_samples):
    #         for j in range(self.nb_samples):
    #             for l in range(self.latent_dim):
    #                 delta[i, j] += self.delta(Z[i, l], Z[j, l])
    #
    #     s = 0
    #     for i in range(self.nb_samples):
    #         for j in range(self.nb_samples):
    #             s += (r[j, i]*dT[j, j]*d[j, i] + r[i, j]*dT[i, i]*d[i, j])*delta[j, i]
    #
    #     return 2*s/(self.nb_samples*self.sigma**2)
    def dEdZ(self, Z, X, alpha=0.008, norm=4):
        dz = np.sum((Z[:, None, :] - Z[None, :, :])**2, axis=2)
        k = np.exp(-1*dz/(2*self.sigma**2))
        r = k/np.sum(k, axis=1, keepdims=True)
        delta = Z[:, None, :] - Z[None, :, :]
        d = self.f(Z, Z)[:, None, :] - X[None, :, :]
        dt = self.f(Z, Z)[None, :, :] - X[:, None, :]

        R = np.einsum("ni,nnd,nid->ni", r, dt, d) + np.einsum("in,iid,ind->in", r, dt, d)
        R = 2*(np.einsum("ni,nil->nl", R, delta))/(self.nb_samples*self.sigma**2)

        #l = np.sum((r[:, :, None, None] * dt[:, None, :, :])[:, :, None, :] * d[:, None, :, :], axis=(1,3))
        #l2 = np.einsum("ni,nnd,nid->ni", r, dt, d)
        #print(np.allclose(l, l2))
        return R

    def fit(self, nb_epoch: int, eta: float):
        # 学習過程記録用
        self.history['z'] = np.zeros((nb_epoch, self.nb_samples, self.latent_dim))
        self.history['f'] = np.zeros((nb_epoch, self.nb_samples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
            # Zの更新
            self.Z = self.Z - eta*jax.grad(self.E, argnums=0)(self.Z, self.X)/self.nb_samples
            #self.Z = self.Z - eta*self.dEdZ(self.Z, self.X)/self.nb_samples
            # 学習過程記録用
            self.history['z'][epoch] = self.Z
            self.history['f'][epoch] = self.f(self.Z, self.Z)
            self.history['error'][epoch] = self.E(self.Z, self.X)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_f(self, resolution): #fのメッシュ描画用，resolution:一辺の代表点の数
         nb_epoch = self.history['z'].shape[0]
         self.history['y'] = np.zeros((nb_epoch, resolution ** self.latent_dim, self.ob_dim))
         #self.history['y'] = np.zeros((nb_epoch, self.X.shape[0], self.ob_dim))
         for epoch in tqdm(range(nb_epoch)):
             zeta = create_zeta_2D(self.history['z'][epoch], resolution)
             #zeta = create_zeta_1D(self.history['z'][epoch])
             Y = self.f(zeta, self.history['z'][epoch])
             self.history['y'][epoch] = Y
         return self.history['y']

    def d(self, x, y):
        return y - x

    def delta(self, Zi, Zj):
        return Zi - Zj

    def k(self, Z, Zi):
        d = np.zeros(Zi.shape[0])
        k = np.zeros(Zi.shape[0])
        for i in range(Zi.shape[0]):
            for j in range(Zi.shape[1]):
                d[i] += (self.delta(Z[j], Zi[i, j])) ** 2
            k[i] = np.exp(-1 * d[i] / (2 * self.sigma ** 2))

        return k

    def K(self, Z, Zi):
        k = self.k(Z, Zi)
        K = np.sum(k)

        return K

def create_zeta_2D(Z, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
    #XX, YY = np.meshgrid(np.linspace(-Z/400, Z/400, resolution), np.linspace(-Z/400, Z/400, resolution))
    zmax = np.amax(Z, axis=0)
    zmin = np.amin(Z, axis=0)
    XX, YY = np.meshgrid(np.linspace(zmin[0], zmax[0], resolution), np.linspace(zmin[1], zmax[1], resolution))
    xx = XX.reshape(-1)
    yy = YY.reshape(-1)
    zeta = np.concatenate([xx[:, None], yy[:, None]], axis=1)

    return zeta

def create_zeta_1D(Z):
    zmax = np.amax(Z)
    zmin = np.amin(Z)
    zeta = np.linspace(zmin, zmax, Z.shape[0]).reshape(-1, 1)

    return zeta

if __name__ == '__main__':
    from Lecture_UKR.data import create_kura
    from Lecture_UKR.data import create_rasen
    from Lecture_UKR.tokunaga.data import create_2d_sin_curve
    from Lecture_UKR.tokunaga.data import create_big_kura
    from Lecture_UKR.tokunaga.data import create_cluster
    from Lecture_UKR.tokunaga.load import load_data
    from visualizer import visualize_history
    from visualizer import visualize_real_history

    #各種パラメータ変えて遊んでみてね．
    epoch = 300 #学習回数
    sigma = 0.22 #カーネルの幅
    eta = 50 #学習率
    latent_dim = 1 #潜在空間の次元

    seed = 20
    np.random.seed(seed)


    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples = 100 #データ数
    #X = create_kura(nb_samples) #鞍型データ　ob_dim=3, 真のL=2
    #X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    #X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1
    #X = create_big_kura(nb_samples)
    #X = create_cluster(nb_samples)
    X = load_data()[0]
    ukr = UKR(X, latent_dim, sigma, prior='random')
    ukr.fit(epoch, eta)
    visualize_real_history(load_data(), ukr.history['z'], ukr.history['error'], save_gif=True, filename="seed20")
    #visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")

    #----------描画部分が実装されたらコメントアウト外す----------
    #ukr.calc_approximate_f(resolution=10)
    #visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")



