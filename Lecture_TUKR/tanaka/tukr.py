import numpy as np
import jax,jaxlib
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm #プログレスバーを表示させてくれる
from sklearn.datasets import load_iris



class TUKR:
    def __init__(self, X, nb_samples1, nb_samples2, latent_dim1, latent_dim2, sigma, prior='random', Uinit=None, Vinit=None):
        #--------初期値を設定する．---------
        self.X = X
        #ここから下は書き換えてね

        if X.ndim == 3:
            self.nb_samples1, self.nb_samples2, self.ob_dim = self.X.shape
        else:
            self.nb_samples1, self.nb_samples2 = self.X.shape
            self.ob_dim = 1
            self.X = X[:,:,None]

        self.sigma = sigma
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.alpha = alpha
        self.norm = norm

        if Uinit is None:
            if prior == 'random': #一様事前分布のとき
                self.U = np.random.normal(0, 0.1 * self.sigma, size=(self.nb_samples1, self.latent_dim1))
                #(平均,標準偏差,配列のサイズ)
            # else: #ガウス事前分布のとき
            #     U =
        else: #Zの初期値が与えられた時
            self.U = Uinit

        self.history = {}

        if Vinit is None:
            if prior == 'random':  # 一様事前分布のとき
                self.V = np.random.normal(0, 0.1 * self.sigma, size=(self.nb_samples2, self.latent_dim2))
                # (平均,標準偏差,配列のサイズ)
            # else: #ガウス事前分布のとき
            #     V =
        else:  # Zの初期値が与えられた時
            self.V = Vinit

        self.history = {}

    def f(self, U, V): #写像の計算
        DistU = jnp.sum((U[:, None, :] - U[None, :, :]) ** 2, axis=2)
        DistV = jnp.sum((V[:, None, :] - V[None, :, :]) ** 2, axis=2)
        HU = jnp.exp((-1 * DistU) / (2 * (self.sigma) ** 2))
        HV = jnp.exp((-1 * DistV) / (2 * (self.sigma) ** 2))
        # GU = jnp.sum(HU, axis=1)[:, None]
        # GV = jnp.sum(HV, axis=1)[:, None]
        # RU = HU / GU
        # RV = HV / GV
        f = jnp.einsum('li,kj,ijd->lkd', HU, HV, self.X)
        f1 = jnp.einsum('li,kj->lk', HU, HV)
        f2 = f1[:, :, None]
        return f / f2

    def ff(self, U, V, epoch): #写像の計算
        DistU = jnp.sum((U[:, None, :] - self.history['u'][epoch][None, :, :]) ** 2, axis=2)
        DistV = jnp.sum((V[:, None, :] - self.history['v'][epoch][None, :, :]) ** 2, axis=2)
        HU = jnp.exp((-1 * DistU) / (2 * (self.sigma) ** 2))
        HV = jnp.exp((-1 * DistV) / (2 * (self.sigma) ** 2))
        # GU = jnp.sum(HU, axis=1)[:, None]
        # GV = jnp.sum(HV, axis=1)[:, None]
        # RU = HU / GU
        # RV = HV / GV
        f = jnp.einsum('li,kj,ijd->lkd', HU, HV, self.X)
        f1 = jnp.einsum('li,kj->lk', HU, HV)
        f2 = f1[:, :, None]
        return f / f2

    def E(self,U,V,X,alpha,norm):#目的関数の計算
        Y = self.f(U,V)
        e = jnp.sum((X - Y) ** 2)
        r = alpha*(jnp.sum(U**norm)+jnp.sum(V**norm))
        e = e/(self.nb_samples1*self.nb_samples2)
        r = r/(self.nb_samples1*self.nb_samples2)
        return e + r

    def fit(self, nb_epoch: int, eta: float,alpha,norm):
        # 学習過程記録用
        self.history['u'] = np.zeros((nb_epoch, self.nb_samples1, self.latent_dim2))
        self.history['v'] = np.zeros((nb_epoch, self.nb_samples2, self.latent_dim1))
        self.history['f'] = np.zeros((nb_epoch, self.nb_samples1, self.nb_samples2, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):

            # U,Vの更新
            dEdu = jax.grad(self.E, argnums=0)(self.U, self.V, self.X, alpha, norm) / self.nb_samples1
            self.U = self.U - eta * dEdu

            dEdv = jax.grad(self.E, argnums=1)(self.U, self.V, self.X, alpha, norm) / self.nb_samples2
            self.V = self.V - eta * dEdv

            # 学習過程記録用
            self.history['u'][epoch] = self.U
            self.history['v'][epoch] = self.V
            self.history['f'][epoch] = self.f(self.U, self.V)
            self.history['error'][epoch] = self.E(self.U, self.V, self.X, alpha, norm)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_f(self, resolution): #fのメッシュ描画用，resolution:一辺の代表点の数
        nb_epoch = self.history['u'].shape[0]
        self.history['y'] = np.zeros((nb_epoch,self.nb_samples1,self.nb_samples2, self.ob_dim))
        for epoch in tqdm(range(nb_epoch)):
            Uzeta = self.create_Uzeta(self.history['u'][epoch],resolution)
            Vzeta = self.create_Vzeta(self.history['v'][epoch],resolution)

            y = self.ff(Uzeta,Vzeta,epoch)
            self.history['y'][epoch] = y

    def create_Uzeta(self, U, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
        Uzeta = np.linspace(np.min(U), np.max(U),self.nb_samples1).reshape(-1,1)

        return Uzeta

    def create_Vzeta(self, V, resolution):  # fのメッシュの描画用に潜在空間に代表点zetaを作る．
        Vzeta = np.linspace(np.min(V), np.max(V),self.nb_samples2).reshape(-1,1)

        return Vzeta


if __name__ == '__main__':
    from Lecture_TUKR.tanaka.animal import load_data
    # from Lecture_TUKR.tanaka.data_scratch_tanaka import load_kura_tsom
    # from Lecture_TUKR.tanaka.data_scratch_tanaka import create_rasen
    # from Lecture_TUKR.tanaka.data_scratch_tanaka import create_2d_sin_curve
    from visualizer_animal import visualize_history

    #各種パラメータ変えて遊んでみてね．
    epoch = 200 #学習回数
    sigma = 0.01 #カーネルの幅
    eta = 10  #学習率
    latent_dim1 = 2 #潜在空間の次元
    latent_dim2 = 2 #潜在空間の次元
    alpha = 0.1
    norm = 2
    seed = 4
    np.random.seed(seed)



    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples1 = 10 #データ数
    nb_samples2 = 20
    data = load_data(retlabel_animal=True, retlabel_feature=True)
    # X = load_iris()
    # X = load_kura_tsom(nb_samples1,nb_samples2) #鞍型データ　ob_dim=3, 真のL=2
    # X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    # X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1

    X = data[0]
    animal_label = data[1]
    feature_label = data[2]

    tukr = TUKR(X, nb_samples1, nb_samples2, latent_dim1, latent_dim2, sigma, prior='random')
    tukr.fit(epoch, eta,alpha,norm)
    # visualize_history(X, tukr.history['f'], tukr.history['u'],tukr.history['v'], tukr.history['error'], save_gif=False, filename="tmp")

    #----------描画部分が実装されたらコメントアウト外す----------
    tukr.calc_approximate_f(resolution=10)
    visualize_history(X, tukr.history['y'], tukr.history['u'],tukr.history['v'], tukr.history['error'],animal_label, feature_label, save_gif=False, filename="tmp")



