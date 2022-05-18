import numpy as np
from tqdm import tqdm #プログ
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

class TUKR:
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

    def f(self, U, V): #写像の計算
        su = jnp.sum((U[:, None, :]-U[None, :, :])**2, axis=2)
        sv = jnp.sum((V[:, None, :]-V[None, :, :])**2, axis=2)
        ku = self.kernel(su)
        kv = self.kernel(sv)
        f = jnp.einsum('ni,ni,nd->nd', ku, kv, self.X)/jnp.sum(ku*kv, axis=1, keepdims=True)
        return f

    def kernel(self, s):
        return np.exp(-1 * s / (2 * self.sigma ** 2))

    def E(self, U, V, X, alpha=0.001, norm=2): #目的関数の計算
        d = ((X-self.f(U, V))**2)/(self.nb_samples*self.nb_samples)
        E = jnp.sum(d)+alpha*(jnp.sum(U**norm)+jnp.sum(V**norm))
        return E

    def fit(self, nb_epoch: int, eta: float):
        # 学習過程記録用
        self.history['z'] = np.zeros((nb_epoch, self.nb_samples, self.latent_dim))
        self.history['f'] = np.zeros((nb_epoch, self.nb_samples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
            # Zの更新
            self.Z = self.Z - eta*jax.grad(self.E, argnums=0)(self.Z, self.X)/self.nb_samples
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
    from Lecture_TUKR.tokunaga.data import load_kura_tsom
    from visualizer import visualize_history
    from visualizer import visualize_real_history

    #各種パラメータ変えて遊んでみてね．
    epoch = 200 #学習回数
    sigma = 0.2 #カーネルの幅
    eta = 50 #学習率
    latent_dim1 = 2 #潜在空間1の次元
    latent_dim2 = 2 #潜在空間2の次元

    seed = 4
    np.random.seed(seed)

    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples1 = 10 #データ数
    nb_samples2 = 20
    X = load_kura_tsom(nb_samples1, nb_samples2) #鞍型データ　ob_dim=3, 真のL=2
    #X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    #X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1
    #X = create_big_kura(nb_samples)
    #X = create_cluster(nb_samples)
    #X = load_data()[0]
    ukr = TUKR(X, latent_dim1, latent_dim2, sigma, prior='random')
    ukr.fit(epoch, eta)
    #visualize_real_history(load_data(), ukr.history['z'], ukr.history['error'], save_gif=True, filename="seed20")
    visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")

    #----------描画部分が実装されたらコメントアウト外す----------
    #ukr.calc_approximate_f(resolution=10)
    #visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")



