import numpy as np
import jax,jaxlib
import jax.numpy as jnp
from tqdm import tqdm #プログレスバーを表示させてくれる
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA #主成分分析
from face_project.load import load_angle_resized_data
from face_project.load import load_angle_resized_same_angle_data
from sklearn.manifold import TSNE
from face_project.load import load_angle_resized_data_TUKR

class UKR:
    def __init__(self, X, latent_dim, sigma, prior='random', Zinit=None):
        #--------初期値を設定する．---------
        self.X = X
        #ここから下は書き換えてね
        self.nb_samples, self.ob_dim = X.shape
        self.sigma =sigma
        self.latent_dim =latent_dim

        if Zinit is None:
            if prior == 'random': #一様事前分布のとき
                self.Z = np.random.uniform(0, self.sigma*0.001, (self.nb_samples, self.latent_dim))
                # Z1_vec = np.random.uniform(low=-1, high=1, size=Z)
                # Z1_colum_vec = np.random.uniform(low=-1, high=1, size=[Z, 1])
            else: #ガウス事前分布のとき
                self.Z = np.random.normal(self.nb_samples * self.latent_dim).reshape(self.nb_samples, self.latent_dim)
        else: #Zの初期値が与えられた時
            self.Z = Zinit

        self.history = {}

    def kernel(self, Z1, Z2): #写像の計算
            Mom = jnp.sum((Z1[:, None, :] - Z2[None, :, :]) ** 2, axis=2)
            Chi = jnp.exp(-1/(2*self.sigma**2)*Mom)
            f = (Chi@self.X)/jnp.sum(Chi, axis=1, keepdims=True)

            return f

    def E(self, Z, X, alpha, norm): #目的関数の計算
        E = np.sum((X - self.kernel(Z,Z))**2)
        R = alpha * jnp.sum(jnp.abs(Z ** norm))
        E = E / self.nb_samples + R / self.nb_samples

        return E

    def fit(self, nb_epoch: int, eta: float, alpha: float, norm: float) :
        # 学習過程記録用
        self.history['z'] = np.zeros((nb_epoch, self.nb_samples, self.latent_dim))
        self.history['kernel'] = np.zeros((nb_epoch, self.nb_samples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
            # Zの更新
            dEdx = jax.grad(self.E, argnums=0)(self.Z, self.X, alpha, norm)
            self.Z -= (eta) * dEdx


            # 学習過程記録用
            self.history['z'][epoch] = self.Z
            self.history['kernel'][epoch] = self.kernel(self.Z,self.Z)
            self.history['error'][epoch] = self.E(self.Z,self.X, alpha, norm)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_f(self, resolution): #fのメッシュ描画用，resolution:一辺の代表点の数
        nb_epoch = self.history['z'].shape[0]
        self.history['y'] = np.zeros((nb_epoch, resolution ** self.latent_dim, self.ob_dim))
        for epoch in tqdm(range(nb_epoch)):
            create_zeta = [None, create_zeta_1D, create_zeta_2D][self.latent_dim]
            zeta = create_zeta(self.Z, resolution)
            Y = self.kernel(zeta, self.history['z'][epoch])
            self.history['y'][epoch] = Y
        return self.history['y']

    def calc_approximate_fig(self, resolution): #fのメッシュ描画用，resolution:一辺の代表点の数
        nb_epoch = self.history['z'].shape[0]
        self.history['y'] = np.zeros((nb_epoch, resolution ** self.latent_dim, self.ob_dim))
        for epoch in tqdm(range(nb_epoch)):
            create_zeta = [None, create_zeta_1D, create_zeta_2D][self.latent_dim]
            zeta = create_zeta(self.Z, resolution)
            Y = self.kernel(zeta, self.history['z'][epoch])
            self.history['y'][epoch] = Y
        return self.history['y']


def create_zeta_1D(Z, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
    z_x = np.linspace(np.min(Z), np.max(Z), resolution).reshape(-1, 1)

    return z_x


def create_zeta_2D(Z, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
    z_x = np.linspace(np.min(Z), np.max(Z), resolution)
    z_y = np.linspace(np.min(Z), np.max(Z), resolution)
    XX, YY = np.meshgrid(z_x, z_y)
    xx = XX.reshape(-1)
    yy = YY.reshape(-1)
    zeta = np.concatenate([xx[:, None], yy[:, None]], axis=1)

    return zeta







if __name__ == '__main__':
    from face_project.load import load_angle_resized_data

    from face_project.load import load_angle_resized_same_angle_data
    # from Lecture_UKR.data import create_rasen
    # from Lecture_UKR.data import create_2d_sin_curve
    from UKR_visualizer import visualize_history
    from UKR_visualizer import visualize_new_history
    from face_project.ayukawaa.PCA_ayu import PCA_1

    #各種パラメータ変えて遊んでみてね．
    ##
    epoch = 500 #学習回数
    sigma = 0.2 #カーネルの幅
    eta = 0.0000001 #学習率
    latent_dim = 2 #潜在空間の次元
    alpha = 0.000000001
    norm = 10
    seed = 2
    jedi = 3 #PCAの次元
    r = 5
    np.random.seed(seed)

    pca = PCA(n_components = jedi)  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言 n_componentsで主成分数を定義
    # x = load_angle_resized_data()
    x = load_angle_resized_same_angle_data()
    df = x.reshape(x.shape[0], -1)
    # pca.fit(load_angle_resized_data)# PCA を実行
    # PCA_ans = pca.transform(load_angle_resized_data)
    pca.fit(df)
    kiyo = pca.explained_variance_ratio_
    PCA_ans = pca.transform(df)


    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples = 200 #データ数
    X = PCA_ans # 鞍型データ　ob_dim=3, 真のL=2
    # X = PCA_1() #鞍型データ　ob_dim=3, 真のL=2
    # X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    # X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1

    ukr = UKR(X, latent_dim, sigma, prior='random')
    ukr.fit(epoch, eta, alpha, norm)
    #visualize_history(X, ukr.history['kernel'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")
    #----------描画部分が実装されたらコメントアウト外す----------
    ukr.calc_approximate_f(resolution=200)
    # visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")
    visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/UKR顔tsne")
    # visualize_new_history(X, ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")



    # pca = PCA(n_components = jedi)  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言 n_componentsで主成分数を定義
    # x = load_angle_resized_data()
    # # x = load_angle_resized_same_angle_data()
    # df = x.reshape(x.shape[0], -1)
    # # pca.fit(load_angle_resized_data)# PCA を実行
    # # PCA_ans = pca.transform(load_angle_resized_data)
    # pca.fit(df)
    # kiyo = pca.explained_variance_ratio_
    # PCA_ans = pca.transform(df)
    kiyo_goukei = np.add.accumulate(kiyo)

    ruisekikiyo = np.hstack([0, kiyo.cumsum()])
    print(PCA_ans.shape)
    print(kiyo_goukei)
    print()
    print(ruisekikiyo)





    Y = ukr.calc_approximate_fig(resolution = r**2)
    # Y_inv =pca.inverse_transform(Y)
    # print(Y_inv.shape)

    Y_inv = pca.inverse_transform(Y)
    fig = plt.figure(figsize=(10, 10), dpi = 80)
    gs = fig.add_gridspec(r, r)
    # print(Y_inv.shape)
    # print()
    # print(Y.shape)
    # exit()
    for i in range(r**2):
        # Y_inv = pca.inverse_transform(ukr.calc_approximate_f(resolution=200)[i])
        # Y_inv = pca.inverse_transform(ukr.history["y"][i])
        fig.add_subplot(gs[i // r, i % r])
        img = Y_inv[epoch-1, i, :]
        # print(img.shape)
        # exit()
        img = img.reshape(64, 64)

        plt.imshow(img, cmap='gray')

    plt.show()