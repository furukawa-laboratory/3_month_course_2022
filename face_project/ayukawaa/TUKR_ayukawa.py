import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm #プログレスバーを表示させてくれる
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA #主成分分析
from face_project.load import load_angle_resized_data
from face_project.load import load_angle_resized_same_angle_data
from sklearn.manifold import TSNE
from face_project.load import load_angle_resized_data_TUKR

class TUKR:
    def __init__(self, X, xlatent_dim, ylatent_dim, xsigma, ysigma, prior='random', Zinit=None):
        #--------初期値を設定する．---------
        # self.X = X[0][ :, :, None]
        self.X = X
        # exit()
        print(self.X.shape)
        #ここから下は書き換えてね
        self.nb_xsamples= self.X.shape[0]
        self.nb_ysamples
        self.ob_dim

        self.xsigma, self.ysigma, = xsigma, ysigma
        self.xlatent_dim, self.ylatent_dim = xlatent_dim, ylatent_dim


        # print(self.nb_xsamples, self.nb_ysamples, self.ob_dim)
        if Zinit is None:
            if prior == 'random': #一様事前分布のとき
                self.U = np.random.uniform(low=-0.001, high=0.001, size=(self.nb_xsamples, self.xlatent_dim))
            # self.Z = np.random.uniform(0, self.xsigma * 0.001, (self.nb_xsamples, self.xlatent_dim))

            else: #ガウス事前分布のとき
                self.U = np.random.normal(0, self.xsigma, (self.nb_xsamples, self.xlatent_dim))

        else:  # Zの初期値が与えられた時
            self.Z = Zinit
        self.history = {}

        if Vinit is None:
            if prior == 'random':  # 一様事前分布のとき
                self.V = np.random.uniform(low=-0.001, high=0.001, size=(self.ysamples, self.latent_dim))

            else:  # ガウス事前分布のとき
                self.V = np.random.normal(self.ysamples * self.latent_dim).reshape(self.ysamples, self.latent_dim)
        else:  # Zの初期値が与えられた時
            self.V = Vinit

        self.history = {}

                # self.Z = np.random.uniform(0, self.xsigma * 0.001, (self.nb_xsamples, self.xlatent_dim))
                # self.v = np.random.uniform(0, self.ysigma * 0.001, (self.nb_ysamples, self.ylatent_dim))

                # Z1_vec = np.random.uniform(low=-1, high=1, size=Z)
                # Z1_colum_vec = np.random.uniform(low=-1, high=1, size=[Z, 1])
            # else: #ガウス事前分布のとき
                # else: #Zの初期値が与えられた時
            #self.Z = Zinit

        self.history = {}

    def kernel(self, z1, z2, v1, v2): #写像の計算 TUKRの式に変更
        u_u = jnp.sum((z1[:, None, :] - z2[None, :, :]) ** 2, axis=2)
        v_v = jnp.sum((v1[:, None, :] - v2[None, :, :]) ** 2, axis=2)
        ku_u = jnp.exp(-1/(2*self.xsigma**2)*u_u)#(20,20)
        kv_v = jnp.exp(-1/(2*self.ysigma**2)*v_v)#(10,10)

        # granMa = jnp.sum(ku_u, axis=1, keepdims=True)#(20,1,1)
        # granFa = jnp.sum(kv_v, axis=0, keepdims=True)#(1,10,1)
        # f = (granMa * granFa * self.X)/(granMa * granFa)
        # f = ((Chi@self.X)*(Chi@self.X))/jnp.sum(Chiu, axis=1, keepdims=True)*jnp.sum(Childv, axis=1, keepdims=True)
        # print(ku_u.shape,kv_v.shape,self.X.shape)
        f_no_ue = jnp.einsum('ui, vj, ijd -> uvd', ku_u, kv_v, self.X) #(20,10,3)
        f_no_sh = jnp.einsum('ui, vj -> uv', ku_u, kv_v) #(20,10)
        # print(granMa.shape)
        # print(granPa.shape)
        f = f_no_ue/f_no_sh[:, :, None]
        # print(u_u.shape)
        return f

    def E(self, Z, v, X, alpha, norm): #目的関数の計算
        #print(X.shape)#(20,10,3)　jnp.abs(Z ** norm)(20,2)
        #print(self.kernel(Z, Z, v, v).shape)#(20,10,3)
        E1 = np.sum((X - self.kernel(Z, Z, v, v))**2)
        # E1 = jnp.sum((X - 2) ** 2)
        R_u = np.sum(jnp.abs(Z ** norm))
        R_v = np.sum(jnp.abs(v ** norm))
        E = E1 / (self.nb_xsamples * self.nb_ysamples) + alpha * (R_u + R_v)
        # print(((X - self.kernel(Z, Z, v, v))).shape)
        # print(E1.shape)
        return E

    def fit(self, nb_epoch: int, eta: float, alpha: float, norm: float) :
        # 学習過程記録用
        self.history['z'] = np.zeros((nb_epoch, self.nb_xsamples, self.xlatent_dim))
        self.history['v'] = np.zeros((nb_epoch, self.nb_ysamples, self.ylatent_dim))
        self.history['kernel'] = np.zeros((nb_epoch, self.nb_xsamples, self.nb_ysamples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
            # Zの更新
            dEdx =jax.grad(self.E, argnums=0)(self.Z, self.v, self.X, alpha, norm)
            self.Z -= (eta) * dEdx
            dEdy = jax.grad(self.E, argnums=1)(self.Z, self.v, self.X, alpha, norm)
            self.v -= (eta) * dEdy

            # dEdx = jax.grad(self.E, argnums=0)(self.Z, self.v, self.X, alpha, norm)/self.nb_xsamples
            # self.Z -= (eta) * dEdx
            # dEdy = jax.grad(self.E, argnums=1)(self.Z, self.v, self.X, alpha, norm)/self.nb_ysamples
            # self.v -= (eta) * dEdy
            # 学習過程記録用
            self.history['z'][epoch] = self.Z
            self.history['v'][epoch] = self.v
            self.history['kernel'][epoch] = self.kernel(self.Z, self.Z, self.v, self.v)
            self.history['error'][epoch] = self.E(self.Z, self.v, self.X, alpha, norm)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_f(self, resolution, nb_epoch): #fのメッシュ描画用，resolution:一辺の代表点の数

        self.history['x'] = np.zeros((nb_epoch, resolution ** self.xlatent_dim, resolution ** self.ylatent_dim, self.ob_dim))
        # AA = self.history['z'].shape[0]
        # self.history['x'] = np.zeros((AA, self.nb_xsamples, self.nb_ysamples, self.ob_dim))
        # print(resolution, self.Z.shape)
        for epoch in tqdm(range(nb_epoch)):
            zeta_Z = self.create_zeta(self.Z, resolution)
            zeta_v = self.create_zeta(self.v, resolution)
            X_c = self.kernel(zeta_Z, self.history['z'][epoch], zeta_v, self.history['v'][epoch])
            # X_c = self.f(self.create_zeta(self.history['z'][epoch], resolution), self.Z)
            self.history['x'][epoch] = X_c
        return self.history['x']


    def create_zeta(self,Z, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
        # print(Z)
        # print(resolution)
        z_x = np.linspace(np.min(Z), np.max(Z), resolution).reshape(-1, 1)
        z_y = np.linspace(np.min(Z), np.max(Z), resolution)
        XX, YY = np.meshgrid(z_x, z_y)
        xx = XX.reshape(-1)
        yy = YY.reshape(-1)
        if self.xlatent_dim == 1:
            zeta = z_x
        else:
            zeta = np.concatenate([xx[:, None], yy[:, None]], axis=1)

        return zeta


if __name__ == '__main__':
    # from Lecture_TUKR.ayukawafile.data_scratch import load_kura_tsom
    # from Lecture_TUKR.ayukawafile.visualizer_kura import visualize_history    #kura

    # from Lecture_TUKR.ayukawafile.animals import load_data
    # from Lecture_TUKR.ayukawafile.visualizer import visualize_history  #animal

    from UKR_visualizer import visualize_history
    from face_project.load import load_angle_resized_same_angle_data
    # from face_project.load import load_angle_resized_data_TUKR
    from face_project.ayukawaa.PCA_ayu import PCA_1

    #各種パラメータ変えて遊んでみてね．
    ##
    epoch = 300 #学習回数
    xsigma = 0.3 #カーネルの幅 フィッティングの強度のイメージ　小さいほどその点が持つ引力？が強くなる
    ysigma = 0.3 # カーネルの幅
    eta = 0.5 #学習率 小さい方がゆっくり学習が進む
    xlatent_dim = 1 #潜在空間の次元  鞍型データ用
    ylatent_dim = 1  # 潜在空間の次元

    # xlatent_dim = 2  # 潜在空間の次元
    # ylatent_dim = 2  # 潜在空間の次元  animal
    alpha = 0.0001
    norm = 10
    seed = 2
    jedi = 30 #PCAの次元
    r = 4
    np.random.seed(seed)

    # print(load_angle_resized_data_TUKR().shape)
    # print(load_angle_resized_data_TUKR())


    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_xsamples = load_angle_resized_data_TUKR().shape[0] #データ数
    nb_ysamples = load_angle_resized_data_TUKR().shape[1]

    # print(nb_xsamples)
    # print()
    # print(nb_ysamples)


    # print(TUKR.history['x'].shape)

    pca = PCA(n_components = jedi)  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言 n_componentsで主成分数を定義
    # x = load_angle_resized_data()
    x = load_angle_resized_data_TUKR()
    df = x.reshape(x.shape[0], -1)#(90,33,64,64)
    print(df)
    # pca.fit(load_angle_resized_data)# PCA を実行
    # PCA_ans = pca.transform(load_angle_resized_data)
    pca.fit(df)
    kiyo = pca.explained_variance_ratio_
    PCA_ans = pca.transform(df)


    X = PCA_ans #鞍型データ　ob_dim=3, 真のL=2
    # X = load_data(nb_xsamples, nb_ysamples)   #animal

#(self, X, xlatent_dim, ylatent_dim, xsigma, ysigma, prior='random', Zinit=None):
    ukr = TUKR(X, xlatent_dim, ylatent_dim, xsigma, ysigma, prior='random')
    ukr.fit(epoch, eta, alpha, norm)
    # visualize_history(X, ukr.history['kernel'], ukr.history['z'], ukr.history['v'], ukr.history['error'], save_gif=False, filename="tmp")
    #----------描画部分が実装されたらコメントアウト外す----------
    # print(X.shape)
    ukr.calc_approximate_f(15, epoch)
    #animal
    # visualize_history(X[0][:, :, None], ukr.history['x'], ukr.history['z'], ukr.history['v'], ukr.history['error'], X, save_gif=False, filename="tmp")
    #kura
    visualize_history(X, ukr.history['x'], ukr.history['z'], ukr.history['v'], ukr.history['error'], save_gif=False, filename="tmp")
