import numpy as np
from tqdm import tqdm #プログ
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

class TUKR:
    def __init__(self, X, nb_samples1, nb_samples2, latent_dim1, latent_dim2, sigma, prior='random', Uinit=None, Vinit=None):
        #--------初期値を設定する．---------
        self.X = X
        #ここから下は書き換えてね
        self.nb_samples1, self.nb_samples2, self.ob_dim = self.X.shape
        self.sigma = sigma
        self.latent_dim1, self.latent_dim2 = latent_dim1, latent_dim2

        if Uinit is None:
            if prior == 'random': #一様事前分布のとき
                U = np.random.uniform(-0.01, 0.01, self.nb_samples1*self.latent_dim1).reshape(self.nb_samples1, self.latent_dim1)
            else: #ガウス事前分布のとき
                U = np.random.normal(0, 0.001, self.nb_samples1*self.latent_dim1).reshape(self.nb_samples1, self.latent_dim1)
        else: #Zの初期値が与えられた時
            U = Uinit

        if Vinit is None:
            if prior == 'random':
                V = np.random.uniform(-0.001, 0.001, self.nb_samples2 * self.latent_dim2).reshape(self.nb_samples2,
                                                                                                  self.latent_dim2)
            else:
                V = np.random.normal(0, 0.001, self.nb_samples2 * self.latent_dim2).reshape(self.nb_samples2,
                                                                                            self.latent_dim2)
        else:
            V = Vinit

        self.U = U
        self.V = V
        self.history = {}

    def f(self, U, V): #写像の計算
        su = jnp.sum((U[:, None, :]-self.U[None, :, :])**2, axis=2)
        sv = jnp.sum((V[:, None, :]-self.V[None, :, :])**2, axis=2)
        ku = self.kernel(su)
        kv = self.kernel(sv)
        f = jnp.einsum('il,kj,ijd->ijd', ku, kv, self.X)/jnp.einsum('ii,jj->ij', ku, kv).reshape(nb_samples1, nb_samples2, 1)
        return f

    def kernel(self, d):
        return jnp.exp(-1 * d / (2 * self.sigma ** 2))

    def E(self, U, V, X, alpha=0.001, norm=2): #目的関数の計算
        d = ((X-self.f(U, V))**2)/(self.nb_samples1*self.nb_samples2)
        E = jnp.sum(d)+alpha*(jnp.sum(U**norm)+jnp.sum(V**norm))
        return E

    def fit(self, nb_epoch: int, ueta: float, veta: float):
        # 学習過程記録用
        self.history['u'] = np.zeros((nb_epoch, self.nb_samples1, self.latent_dim1))
        self.history['v'] = np.zeros((nb_epoch, self.nb_samples2, self.latent_dim2))
        self.history['f'] = np.zeros((nb_epoch, self.nb_samples1, self.nb_samples2, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
            # U, Vの更新
            self.U = self.U - ueta*jax.grad(self.E, argnums=0)(self.U, self.V, self.X)/self.nb_samples1
            self.V = self.V - veta*jax.grad(self.E, argnums=1)(self.U, self.V, self.X)/self.nb_samples2
            # 学習過程記録用
            self.history['u'][epoch] = self.U
            self.history['v'][epoch] = self.V
            self.history['f'][epoch] = self.f(self.U, self.V)
            self.history['error'][epoch] = self.E(self.U, self.V, self.X)

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
    ueta = 50 #uの学習率
    veta = 50 #vの学習率
    latent_dim1 = 2 #潜在空間1の次元
    latent_dim2 = 2 #潜在空間2の次元

    seed = 4
    np.random.seed(seed)

    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples1 = 10 #潜在空間１のデータ数
    nb_samples2 = 20 #潜在空間２のデータ数
    X = load_kura_tsom(nb_samples1, nb_samples2)[0] #鞍型データ　ob_dim=3, 真のL=2
    #X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    #X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1
    #X = create_big_kura(nb_samples)
    #X = create_cluster(nb_samples)
    #X = load_data()[0]
    ukr = TUKR(X, nb_samples1, nb_samples2, latent_dim1, latent_dim2, sigma, prior='random')
    ukr.fit(epoch, ueta, veta)
    #visualize_real_history(load_data(), ukr.history['z'], ukr.history['error'], save_gif=True, filename="seed20")
    visualize_history(X, ukr.history['f'], ukr.history['u'], ukr.history['v'], ukr.history['error'], save_gif=True, filename="tmp")

    #----------描画部分が実装されたらコメントアウト外す----------
    #ukr.calc_approximate_f(resolution=10)
    #visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")



