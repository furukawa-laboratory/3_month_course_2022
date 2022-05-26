import numpy as np
import jax,jaxlib
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm #プログレスバーを表示させてくれる


class UKR:
    def __init__(self, X, latent_dim, sigma, prior='random', Zinit=None):
        #--------初期値を設定する．---------
        self.X = X
        #ここから下は書き換えてね
        self.nb_samples, self.ob_dim = X.shape
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.norm = norm

        if Zinit is None:
            if prior == 'random': #一様事前分布のとき
                self.Z = np.random.normal(0, 0.1 * self.sigma, size=(self.nb_samples, self.latent_dim))
                #(平均,標準偏差,配列のサイズ)
            # else: #ガウス事前分布のとき
            #     Z =
        else: #Zの初期値が与えられた時
            self.Z = Zinit

        self.history = {}

    def f(self, Z1, Z2): #写像の計算
        Dist = jnp.sum((Z1[:, None, :] - Z2[None, :, :])**2, axis=2)
        H = jnp.exp((-1* Dist)/(2*(self.sigma)**2))
        G = jnp.sum(H, axis=1)[:, None]
        R = H / G
        f = R @ self.X
        return f

    #def E(self,Z,X,alpha=1,norm=2):
    def E(self,Z,X,alpha,norm):#目的関数の計算
        Y = self.f(Z,Z)

        # e = jnp.sum((X - Y) ** 2)/self.nb_samples
        # e = (1/self.nb_samples) * jnp.sum((X - Y) ** 2)
        e = jnp.sum((X - Y) ** 2)
        r = alpha*jnp.sum(Z**norm)
        e = e/self.nb_samples
        r = r/self.nb_samples
        return e + r

    def fit(self, nb_epoch: int, eta: float):
        # 学習過程記録用
        self.history['z'] = np.zeros((nb_epoch, self.nb_samples, self.latent_dim))
        self.history['f'] = np.zeros((nb_epoch, self.nb_samples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):

            dEdx = jax.grad(self.E,argnums=0)(self.Z,self.X,self.alpha,self.norm)
            self.Z = self.Z - (eta) * dEdx

           # Zの更新



            # 学習過程記録用
            self.history['z'][epoch] = self.Z
            self.history['f'][epoch] = self.f(self.Z,self.Z)
            self.history['error'][epoch] = self.E(self.Z,self.X,self.alpha,self.norm)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_f(self, resolution): #fのメッシュ描画用，resolution:一辺の代表点の数
        nb_epoch = self.history['z'].shape[0]
        self.history['y'] = np.zeros((nb_epoch, resolution ** self.latent_dim, self.ob_dim))
        for epoch in tqdm(range(nb_epoch)):

            y = self.f(self.create_zeta(self.history['z'][epoch],resolution),self.Z)
            self.history['y'][epoch] = y

        return self.history['y']
    def create_zeta(self, Z, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
        a = np.linspace(np.min(Z), np.max(Z), resolution)
        b = np.linspace(np.min(Z), np.max(Z), resolution)
        A,B = np.meshgrid(a,b)
        # A = np.meshgrid(a)
        aa = A.reshape(-1)
        bb = B.reshape(-1)
        zeta = np.concatenate([aa[:,None],bb[:,None]],axis=1)
        #zeta = np.concatenate(aa[:,None],axis=0)

        return zeta


if __name__ == '__main__':
    from Lecture_UKR.tanaka.data import create_kura
    # from Lecture_UKR.tanaka.data import create_rasen
    # from Lecture_UKR.tanaka.data import create_2d_sin_curve
    from visualizer import visualize_history

    #各種パラメータ変えて遊んでみてね．
    epoch = 200 #学習回数
    sigma = 0.03 #カーネルの幅
    eta = 0.1  #学習率
    latent_dim = 2 #潜在空間の次元
    alpha = 0.1
    norm = 2
    seed = 3
    resolution = 100
    np.random.seed(seed)



    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples = 100 #データ数
    X = create_kura(nb_samples) #鞍型データ　ob_dim=3, 真のL=2
    # X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    # X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1

    ukr = UKR(X, latent_dim, sigma, prior='random')
    ukr.fit(epoch, eta)
    visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="mp4")

    #----------描画部分が実装されたらコメントアウト外す----------
    #ukr.calc_approximate_f(resolution)
    #visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")



