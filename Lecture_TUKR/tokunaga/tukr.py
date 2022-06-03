import numpy as np
from tqdm import tqdm #プログ
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit

class TUKR:
    def __init__(self, X, latent_dim1, latent_dim2, sigma1, sigma2, limit, prior='uniform', Uinit=None, Vinit=None):
        #--------初期値を設定する．---------
        self.X = X
        #ここから下は書き換えてね
        self.nb_samples1, self.nb_samples2, self.ob_dim = self.X.shape
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.latent_dim1, self.latent_dim2 = latent_dim1, latent_dim2

        if Uinit is None:
            if prior == 'uniform': #一様事前分布のとき
                U = np.random.uniform(-limit, limit, self.nb_samples1*self.latent_dim1).reshape(self.nb_samples1, self.latent_dim1)
            else: #ガウス事前分布のとき
                U = np.random.normal(0, limit, self.nb_samples1*self.latent_dim1).reshape(self.nb_samples1, self.latent_dim1)
        else: #Zの初期値が与えられた時
            U = Uinit

        if Vinit is None:
            if prior == 'uniform':
                V = np.random.uniform(-limit, limit, self.nb_samples2 * self.latent_dim2).reshape(self.nb_samples2, self.latent_dim2)
            else:
                V = np.random.normal(0, limit, self.nb_samples2 * self.latent_dim2).reshape(self.nb_samples2, self.latent_dim2)
        else:
            V = Vinit

        self.U = U
        self.V = V
        self.history = {}

    def f(self, U, V): #写像の計算
        du = jnp.sum((U[:, None, :]-self.U[None, :, :])**2, axis=2)
        dv = jnp.sum((V[:, None, :]-self.V[None, :, :])**2, axis=2)
        ku = jnp.exp(-1 * du / (2 * self.sigma1 ** 2))
        kv = jnp.exp(-1 * dv / (2 * self.sigma2 ** 2))
        f = jnp.einsum('li,kj,ijd->lkd', ku, kv, self.X)/jnp.einsum('li,kj->lk', ku, kv).reshape(self.nb_samples1, self.nb_samples2, 1)
        return f

    def zetaf(self, U1, U2, V1, V2):
        du = jnp.sum((U1[:, None, :] - U2[None, :, :]) ** 2, axis=2)
        dv = jnp.sum((V1[:, None, :] - V2[None, :, :]) ** 2, axis=2)
        ku = jnp.exp(-1 * du / (2 * self.sigma1 ** 2))
        kv = jnp.exp(-1 * dv / (2 * self.sigma2 ** 2))
        f = jnp.einsum('li,kj,ijd->lkd', ku, kv, self.X) / jnp.einsum('li,kj->lk', ku, kv).reshape(U1.shape[0],
                                                                                                   V1.shape[0], 1)
        return f

    def E(self, U, V, X, alpha=0.001, norm=10): #目的関数の計算
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
            dEdu = jax.grad(self.E, argnums=0)(self.U, self.V, self.X)
            self.U = self.U - ueta*dEdu / self.nb_samples2
            self.V = self.V - veta*jax.grad(self.E, argnums=1)(self.U, self.V, self.X)/self.nb_samples1
            # 学習過程記録用
            self.history['u'][epoch] = self.U
            self.history['v'][epoch] = self.V
            self.history['f'][epoch] = self.f(self.U, self.V)
            self.history['error'][epoch] = self.E(self.U, self.V, self.X)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_f(self, resolutionu, resolutionv): #fのメッシュ描画用，resolution:一辺の代表点の数
         nb_epoch = self.history['u'].shape[0]
         self.history['y'] = np.zeros((nb_epoch, resolutionu ** self.latent_dim1, resolutionv ** self.latent_dim2, self.ob_dim))
         self.history['zetau'] = np.zeros((nb_epoch, resolutionu ** self.latent_dim1, self.latent_dim1))
         self.history['zetav'] = np.zeros((nb_epoch, resolutionv ** self.latent_dim2, self.latent_dim2))
         #self.history['y'] = np.zeros((nb_epoch, self.X.shape[0], self.ob_dim))
         for epoch in tqdm(range(nb_epoch)):
             zetau = create_zeta_2D(self.history['u'][epoch], resolutionu)
             zetav = create_zeta_2D(self.history['v'][epoch], resolutionv)
             Y = self.zetaf(zetau, self.history['u'][epoch], zetav, self.history['v'][epoch])
             self.history['y'][epoch] = Y
             self.history['zetau'][epoch] = zetau
             self.history['zetav'][epoch] = zetav
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

def create_zeta_1D(Z, resolution):
    zmax = np.amax(Z)
    zmin = np.amin(Z)
    zeta = np.linspace(zmin, zmax, Z.shape[0]).reshape(-1, 1)

    return zeta

if __name__ == '__main__':
    from Lecture_TUKR.tokunaga.data import load_kura_tsom
    from Lecture_TUKR.tokunaga.data import load_kura_list
    #from Lecture_TUKR.tokunaga.load import load_angle_resized_data
    from Lecture_TUKR.tokunaga.load import load_animal_data
    from Lecture_TUKR.tokunaga.visualizer import visualize_real_history
    from visualizer import visualize_history

    #各種パラメータ変えて遊んでみてね．
    epoch = 350 #学習回数
    limit = 0.01
    #sigma1 = 2 #uのカーネルの幅
    #sigma2 = 3 #vのカーネル幅
    ueta = 40 #uの学習率
    veta = 20 #vの学習率
    latent_dim1 = 2 #潜在空間1の次元
    latent_dim2 = 2 #潜在空間2の次元

    seed = 4
    np.random.seed(seed)

    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples1 = 10 #潜在空間１のデータ数
    nb_samples2 = 20 #潜在空間２のデータ数
    sigma1 = np.log(nb_samples1)/nb_samples1
    sigma2 = np.log(nb_samples2)/nb_samples2
    #X = load_kura_tsom(nb_samples1, nb_samples2) #鞍型データ　ob_dim=3, 真のL=2
    #X = load_kura_list(nb_samples1, nb_samples2) #record型の蔵型データ ob_dom=3, L=2
    #X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    #X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1
    #X = create_big_kura(nb_samples)
    #X = create_cluster(nb_samples)
    X = load_animal_data()[0][:, :, None]
    #X = load_angle_resized_data()
    tukr = TUKR(X, latent_dim1, latent_dim2, sigma1, sigma2, limit, prior='uniform')
    #print(tukr.list_f(tukr.U, tukr.V))
    tukr.fit(epoch, ueta, veta)
    # np.save('u_history', tukr.history['u'][-1])
    # np.save('v_history', tukr.history['v'][-1])
    visualize_real_history(load_animal_data(), tukr.history['u'], tukr.history['v'], tukr.history['error'], save_gif=False, filename="seed20")
    # visualize_history(X, tukr.history['f'], tukr.history['u'], tukr.history['v'], tukr.history['error'], save_gif=False, filename="iikanzi")

    #----------描画部分が実装されたらコメントアウト外す----------
    #tukr.calc_approximate_f(resolutionu=50, resolutionv=50)
    # np.save('zetau_history', tukr.history['zetau'][-1])
    # np.save('zetav_history', tukr.history['zetav'][-1])
    # np.save('Y_history', tukr.history['y'][-1])
    #visualize_real_history(X, tukr.history['y'], tukr.history['u'], tukr.history['v'], tukr.history['error'], save_gif=False, filename="random")



