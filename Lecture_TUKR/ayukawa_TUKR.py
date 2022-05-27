import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm #プログレスバーを表示させてくれる


class TUKR:
    def __init__(self, X, xlatent_dim, ylatent_dim, xsigma, ysigma, prior='random', Zinit=None):
        #--------初期値を設定する．---------
        self.X = X
        #ここから下は書き換えてね
        self.nb_xsamples, self.nb_ysamples,self.ob_dim = X.shape
        self.xsigma, self.ysigma, = xsigma, ysigma
        self.xlatent_dim, self.ylatent_dim = xlatent_dim, ylatent_dim

        if Zinit is None:
            if prior == 'random': #一様事前分布のとき
                self.Z = np.random.uniform(0, self.xsigma * 0.001, (self.nb_xsamples, self.xlatent_dim))
                self.v = np.random.uniform(0, self.ysigma * 0.001, (self.nb_ysamples, self.ylatent_dim))
                # Z1_vec = np.random.uniform(low=-1, high=1, size=Z)
                # Z1_colum_vec = np.random.uniform(low=-1, high=1, size=[Z, 1])
            # else: #ガウス事前分布のとき
                # else: #Zの初期値が与えられた時
            #self.Z = Zinit

        self.history = {}

    def kernel(self, z1, z2, v1, v2): #写像の計算 TUKRの式に変更
        u_u = jnp.sum((z1[:, None, :] - z2[None, :, :]) ** 2, axis=2, keepdims=True)
        v_v = jnp.sum((v1[:, None, :] - v2[None, :, :]) ** 2, axis=2, keepdims=True)
        ku_u = jnp.exp(-1/(2*self.xsigma**2)*u_u)#(20,20)
        kv_v = jnp.exp(-1/(2*self.ysigma**2)*v_v)#(10,10)

        # granMa = jnp.sum(ku_u, axis=1, keepdims=True)#(20,1,1)
        # granFa = jnp.sum(kv_v, axis=0, keepdims=True)#(1,10,1)
        # f = (granMa * granFa * self.X)/(granMa * granFa)
        # f = ((Chi@self.X)*(Chi@self.X))/jnp.sum(Chiu, axis=1, keepdims=True)*jnp.sum(Childv, axis=1, keepdims=True)

        f_no_ue = jnp.einsum('ui, vj, ijd -> uvd', ku_u, kv_v, self.X)#(20,10,3)
        f_no_sh = jnp.einsum('ui, vj -> uv', ku_u, kv_v)#(20,10)
        # print(granMa.shape)
        # print(granPa.shape)
        f = f_no_ue/f_no_sh[:, :, None]
        print(u_u.shape)
        return f

    def E(self, Z, v, X, alpha, norm): #目的関数の計算
        #E1 = jnp.sum((X - self.kernel(Z, Z, v, v))**2)
        E1 = jnp.sum((X - 2) ** 2)
        R = alpha * jnp.sum(jnp.abs(Z ** norm))
        E = E1 / self.nb_xsamples + R / self.nb_ysamples

        return E

    def fit(self, nb_epoch: int, eta: float, alpha: float, norm: float) :
        # 学習過程記録用
        self.history['z!'] = np.zeros((nb_epoch, self.nb_xsamples, self.xlatent_dim))
        self.history['v!'] = np.zeros((nb_epoch, self.nb_ysamples, self.ylatent_dim))
        self.history['kernel'] = np.zeros((nb_epoch, self.nb_xsamples, self.nb_ysamples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
            # Zの更新
            dEdx =jax.grad(self.E, argnums=0)(self.Z, self.v, self.X, alpha, norm)
            self.Z -= (eta) * dEdx
            # dEdy = jax.grad(self.E, argnums=0)(self.v, self.X, alpha, norm)
            # self.v -= (eta) * dEdx

            # 学習過程記録用
            self.history['z!'][epoch] = self.Z
            self.history['v!'][epoch] = self.v
            self.history['kernel'][epoch] = self.kernel(self.Z,self.Z,self.v,self.v)
            self.history['error'][epoch] = self.E(self.Z,self.X, alpha, norm)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_f(self, resolution, nb_epoch): #fのメッシュ描画用，resolution:一辺の代表点の数
        self.history['x'] = np.zeros((nb_epoch, resolution ** self.xlatent_dim, self.ob_dim))

        for epoch in tqdm(range(nb_epoch)):
            zeta = create_zeta(self.Z, resolution)
            #X = self.kernel(zeta, self.history['z'][epoch])#
            self.history['x'][epoch] = X
        return self.history['x']


    def create_zeta(Z, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
        z_x = np.linspace(np.min(Z), np.max(Z), resolution).reshape(-1, 1)
        z_y = np.linspace(np.min(Z), np.max(Z), resolution)
        XX, YY = np.meshgrid(z_x, z_y)
        xx = XX.reshape(-1)
        yy = YY.reshape(-1)
        zeta = np.concatenate([xx[:, None], yy[:, None]], axis=1)

        return zeta


if __name__ == '__main__':
    from Lecture_TUKR.data_scratch import load_kura_tsom
    # from Lecture_TUKR import create_rasen
    # from Lecture_TUKR import create_2d_sin_curve
    from Lecture_TUKR.visualizer import visualize_history

    #各種パラメータ変えて遊んでみてね．
    ##
    epoch = 1 #学習回数
    xsigma = 0.4 #カーネルの幅
    ysigma = 0.5  # カーネルの幅
    eta = 2 #学習率
    xlatent_dim = 2 #潜在空間の次元
    ylatent_dim = 2  # 潜在空間の次元

    alpha = 0
    norm = 10

    seed = 2
    np.random.seed(seed)

    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_xsamples = 20 #データ数
    nb_ysamples = 10
    X = load_kura_tsom(nb_xsamples,nb_ysamples) #鞍型データ　ob_dim=3, 真のL=2
    # X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    # X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1
#(self, X, xlatent_dim, ylatent_dim, xsigma, ysigma, prior='random', Zinit=None):
    ukr = TUKR(X, xlatent_dim, ylatent_dim, xsigma, ysigma, prior='random')
    ukr.fit(epoch, eta, alpha, norm)
    #visualize_history(X, ukr.history['kernel'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")
    #----------描画部分が実装されたらコメントアウト外す----------
    ukr.calc_approximate_f(10, epoch)
    # visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")

