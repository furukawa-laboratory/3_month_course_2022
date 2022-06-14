import numpy as np
from tqdm import tqdm #プログ
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit

class List_UKR:
    def __init__(self, X, X_num, nb_samles1, nb_samples2, latent_dim1, latent_dim2, sigma1, sigma2, prior='random', Uinit=None, Vinit=None):
        #--------初期値を設定する．---------
        self.X, X_num = X, X_num
        #ここから下は書き換えてね
        self.nb_samples1, self.nb_samples2 = nb_samles1, nb_samples2
        self.ob_dim = X.shape[1]
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.latent_dim1, self.latent_dim2 = latent_dim1, latent_dim2

        if Uinit is None:
            if prior == 'normal': #一様事前分布のとき
                U = np.random.uniform(-0.001, 0.001, self.nb_samples1 * self.latent_dim1).reshape(self.nb_samples1, self.latent_dim1)
            else: #ガウス事前分布のとき
                U = np.random.normal(0, 0.001, self.nb_samples1 * self.latent_dim1).reshape(self.nb_samples1, self.latent_dim1)
        else: #Zの初期値が与えられた時
            U = Uinit

        if Vinit is None:
            if prior == 'normal':
                V = np.random.uniform(-0.001, 0.001, self.nb_samples2 * self.latent_dim2).reshape(self.nb_samples2, self.latent_dim2)
            else:
                V = np.random.normal(0, 0.001, self.nb_samples2 * self.latent_dim2).reshape(self.nb_samples2, self.latent_dim2)
        else:
            V = Vinit

        self.U = U
        self.V = V
        self.history = {}

    def f(self, U, V): #写像の計算
        NU = U[X_num[:, 0]]
        NV = V[X_num[:, 1]]
        du = jnp.sum((NU[:, None, :]-NU[None, :, :])**2, axis=2)
        dv = jnp.sum((NV[:, None, :]-NV[None, :, :])**2, axis=2)
        ku = jnp.exp(-1 * du / (2 * self.sigma1 ** 2))
        kv = jnp.exp(-1 * dv / (2 * self.sigma2 ** 2))
        f = ku * kv @ self.X / np.sum(ku * kv, axis=1, keepdims=True)
        return f

    def zetaf(self, X, zetaU, U, zetaV, V):
        # NU = U[X_num[:, 0]]
        # NV = V[X_num[:, 1]]
        # zetaNU = zetaU[X_num[:, 0]]
        # zetaNV = zetaV[X_num[:, 1]]
        # du = jnp.sum((zetaNU[:, None, :] - NU[None, :, :]) ** 2, axis=2)
        # dv = jnp.sum((zetaNV[:, None, :] - NV[None, :, :]) ** 2, axis=2)
        # ku = jnp.exp(-1 * du / (2 * self.sigma1 ** 2))
        # kv = jnp.exp(-1 * dv / (2 * self.sigma2 ** 2))
        # f = ku * kv @ self.X / np.sum(ku * kv, axis=1, keepdims=True)
        du = np.sum((zetaU[:, None, :] - U[None, :, :])**2, axis=2)
        dv = np.sum((zetaV[:, None, :] - V[None, :, :])**2, axis=2)
        ku = np.exp(-1 * du / (2 * self.sigma1 ** 2))
        kv = np.exp(-1 * dv / (2 * self.sigma2 ** 2))
        f = np.einsum('li,kj,ijd->lkd', ku, kv, X) / np.einsum('li,kj->lk', ku, kv)[:, :, None]
        return f

    def E(self, U, V, X, alpha=0.01, norm=2): #目的関数の計算
        d = ((X-self.f(U, V))**2)/(X.shape[0])
        E = jnp.sum(d)+alpha*(jnp.sum(U**norm)+jnp.sum(V**norm))
        return E

    def fit(self, nb_epoch: int, ueta: float, veta: float):
        # 学習過程記録用
        self.history['u'] = np.zeros((nb_epoch, self.nb_samples1, self.latent_dim1))
        self.history['v'] = np.zeros((nb_epoch, self.nb_samples2, self.latent_dim2))
        self.history['f'] = np.zeros((nb_epoch, self.X.shape[0], self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
            # U, Vの更新
            self.U = self.U - ueta*jax.grad(self.E, argnums=0)(self.U, self.V, self.X)/self.nb_samples2
            self.V = self.V - veta*jax.grad(self.E, argnums=1)(self.U, self.V, self.X)/self.nb_samples1
            # 学習過程記録用
            self.history['u'][epoch] = self.U
            self.history['v'][epoch] = self.V
            self.history['f'][epoch] = self.f(self.U, self.V)
            self.history['error'][epoch] = self.E(self.U, self.V, self.X)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_f(self, resolution): #fのメッシュ描画用，resolution:一辺の代表点の数
         nb_epoch = self.history['u'].shape[0]
         self.history['y'] = np.zeros((nb_epoch, self.nb_samples1, self.nb_samples2, self.ob_dim))
         self.history['zetau'] = np.zeros((nb_epoch, self.nb_samples1, self.latent_dim1))
         self.history['zetav'] = np.zeros((nb_epoch, self.nb_samples2, self.latent_dim2))
         zetaX = np.zeros((self.nb_samples1, self.nb_samples2, self.ob_dim))
         for n in range(self.X.shape[0]):
             zetaX[X_num[n, 0], X_num[n, 1], :] = X[n, :]
         #self.history['y'] = np.zeros((nb_epoch, self.X.shape[0], self.ob_dim))
         for epoch in tqdm(range(nb_epoch)):
             zetau = [None, create_zeta_1D(self.history['u'][epoch]), create_zeta_2D(self.history['u'], resolution)][self.latent_dim1]
             zetav = [None, create_zeta_1D(self.history['v'][epoch]), create_zeta_2D(self.history['v'], resolution)][self.latent_dim2]
             Y = self.zetaf(zetaX, zetau, self.history['u'][epoch], zetav, self.history['v'][epoch])
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
    from Lecture_TUKR.tokunaga.data import load_kura_list
    #from Lecture_TUKR.tokunaga.load import load_angle_resized_data
    from Lecture_TUKR.tokunaga.l_visualizer import visualize_history
    from Lecture_TUKR.tokunaga.data import load_kura_lost_list

    #各種パラメータ変えて遊んでみてね．
    epoch = 200 #学習回数
    #sigma1 = 2 #uのカーネルの幅
    #sigma2 = 3 #vのカーネル幅
    ueta = 1 #uの学習率
    veta = 2 #vの学習率
    latent_dim1 = 1 #潜在空間1の次元
    latent_dim2 = 1 #潜在空間2の次元

    seed = 4
    np.random.seed(seed)

    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples1 = 10 #潜在空間１のデータ数
    nb_samples2 = 15 #潜在空間２のデータ数
    sigma1 = np.log(nb_samples1)/nb_samples1
    sigma2 = np.log(nb_samples2)/nb_samples2
    #X, X_num = load_kura_list(nb_samples1, nb_samples2) #record型の蔵型データ ob_dom=3, L=2
    X, X_num = load_kura_list(nb_samples1, nb_samples2)
    lukr = List_UKR(X, X_num, nb_samples1, nb_samples2, latent_dim1, latent_dim2, sigma1, sigma2, prior='normal')
    #print(tukr.list_f(tukr.U, tukr.V))
    lukr.fit(epoch, ueta, veta)
    #visualize_real_history(load_data(), ukr.history['z'], ukr.history['error'], save_gif=True, filename="seed20")
    #visualize_history(X, lukr.history['f'], lukr.history['u'], lukr.history['v'], lukr.history['error'], save_gif=False, filename="recode_iikanzi")

    #----------描画部分が実装されたらコメントアウト外す----------
    lukr.calc_approximate_f(resolution=10)
    visualize_history(X, X_num, lukr.history['y'], lukr.history['u'], lukr.history['v'], lukr.history['error'], save_gif=False, filename="record_colormap_dekitenai")


