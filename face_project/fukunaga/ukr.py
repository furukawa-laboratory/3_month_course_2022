import numpy as np
from tqdm import tqdm #プログレスバーを表示させてくれる
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from sklearn import preprocessing
# from face_project.fukunaga.UKR_visualizer import visualize_history
from face_project.fukunaga.UKR_visualizer import visualize_history_obs
from face_project.fukunaga.UKR_visualizer import visualize_PNG_obs
from face_project.fukunaga.UKR_visualizer import visualize_history_no_obs
from face_project.fukunaga.UKR_visualizer import visualize_PNG_no_obs
# from face_project.fukunaga.PCA import x_PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from face_project.fukunaga.load import load_angle_resized_data
from sklearn.manifold import TSNE
from face_project.fukunaga.load import load_angle_resized_same_angle_data
from face_project.fukunaga.load import load_angle_resized_data_TUKR


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
                self.Z = -np.random.normal(0, 0.02, (self.nb_samples, self.latent_dim))
            else: #ガウス事前分布のとき
                self.Z = np.random.normal(self.nb_samples*self.latent_dim).reshape(self.nb_samples, self.latent_dim)
        else: #Zの初期値が与えられた時
            self.Z = Zinit

        self.history = {}
        # print(self.Z)
    def f(self, Z1, Z2):
        d = np.sum((Z1[:, None, :]-Z2[None, :, :])**2, axis=2)
        H = -1*(d/(2*self.sigma**2))
        h = jnp.exp(H)
        # print(d.shape)
        # print(h.shape)
        # print(self.X.shape)
        bunshi = h@self.X
        bunbo = np.sum(h, axis=1, keepdims=True)

        f = bunshi/bunbo

        #写像の計算

        return f

    def E(self, Z, X, alpha, norm): #目的関数の計算
        E = jnp.sum((X - self.f(Z,Z))**2)
        R = alpha*jnp.sum(jnp.abs(Z**norm))
        E = E/self.nb_samples + R/self.nb_samples

        return E

    def fit(self, nb_epoch: int, eta: float, alpha: float, norm: float):
        # 学習過程記録用
        self.history['z'] = np.zeros((nb_epoch, self.nb_samples, self.latent_dim))
        self.history['f'] = np.zeros((nb_epoch, self.nb_samples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in range(nb_epoch):
            self.history['z'][epoch] = self.Z   #初期化状態を見る時
            dEdx = jax.grad(self.E, argnums=0)(self.Z, self.X, alpha, norm)
            self.Z = self.Z -eta * dEdx

           # Zの更新




            # 学習過程記録用
            # self.history['z'][epoch] = self.Z
            self.history['f'][epoch] =self.f(self.Z, self.Z)
            self.history['error'][epoch] =self.E(self.Z, self.X, alpha, norm)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_f(self, resolution): #fのメッシュ描画用，resolution:一辺の代表点の数
        nb_epoch = self.history['z'].shape[0]
        self.history['y'] = np.zeros((nb_epoch, resolution ** self.latent_dim, self.ob_dim))
        for epoch in tqdm(range(nb_epoch)):
            create_zeta = [None, create_zeta_1D, create_zeta_2D][self.latent_dim]
            zeta = create_zeta(self.Z, resolution)
            Y = self.f(zeta, self.history['z'][epoch])
            self.history['y'][epoch] = Y
        return Y


def create_zeta_1D(Z, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
    z_x = np.linspace(np.min(Z), np.max(Z), resolution).reshape(-1, 1)
    # z_x = np.linspace(np.min(Z), np.max(Z), resolution)
    # z_y = np.linspace(np.min(Z), np.max(Z), resolution)
    # XX, YY = np.meshgrid(z_x, z_y)
    # xx = XX.reshape(-1)
    # yy = YY.reshape(-1)
    # zeta = np.concatenate([xx[:, None], yy[:, None]], axis=1)


    return z_x
def create_zeta_2D(Z, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
    # z_x = np.linspace(np.min(Z), np.max(Z), resolution).reshape(-1, 1)
    z_x = np.linspace(np.min(Z), np.max(Z), resolution)
    z_y = np.linspace(np.min(Z), np.max(Z), resolution)
    XX, YY = np.meshgrid(z_x, z_y)
    xx = XX.reshape(-1)
    yy = YY.reshape(-1)
    zeta = np.concatenate([xx[:, None], yy[:, None]], axis=1)


    return zeta

def img(r, latent_dim):
    if latent_dim == 1:
        Y = ukr.calc_approximate_f(resolution=r*r)
    else:
        Y = ukr.calc_approximate_f(resolution=r)
    Y_inv = pca.inverse_transform(Y)
    # print(Y_inv.shape)
    fig = plt.figure(figsize=(10, 10), dpi = 80)
    gs = fig.add_gridspec(r, r)
    for i in range(r**2):
        ax = fig.add_subplot(gs[i // r, i % r])
        img = Y_inv[i, :]
        img = img.reshape(64, 64)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img, cmap='gray')


    plt.show()

if __name__ == '__main__':

    ###########################PCA


    # x, angle = load_angle_resized_data('01')
    x, label = load_angle_resized_same_angle_data('0')
    pca = PCA(n_components=3)
    # print(78789789789)
    # print(x.shape)
    # print(x.reshape(x.shape[0], -1).shape)
    # print(88888888)

    x_2d = pca.fit_transform(x.reshape(x.shape[0], -1))
    # print(x_2d.shape)

    X = x_2d
    # 寄与率
    cr = pca.explained_variance_ratio_
    # 累積寄与率
    ccr = np.add.accumulate(cr)
    print(ccr)
    #各種パラメータ変えて遊んでみてね．
    epoch = 1000 #学習回数
    sigma = 1 #カーネルの幅
    eta = 0.00001#学習率
    latent_dim = 2 #潜在空間の次元
    alpha = 0.00001
    norm = 10
    seed = 20
    np.random.seed(seed)
    r = 10

    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples = 100 #データ数
    #X = x_PCA()
    # X = load_date()[0]

    #########PCA初期化######
    # z, an = load_angle_resized_data('01')
    z,la = load_angle_resized_same_angle_data('0')
    pca_creat = PCA(n_components=1)
    z_ini = pca_creat.fit_transform(z.reshape(z.shape[0], -1))
    # print(z_ini.shape)
    # exit()
    mmscaler = MinMaxScaler(feature_range=(-0.1, 0.1), copy=True)
    mmscaler.fit(z_ini)
    z_ini = mmscaler.transform(z_ini)

    zero = np.zeros((90, 1))
    # print(zero)
    zero = np.random.normal(0, 0.0000000001, (90, 1))
    # lin = np.linspace(-1, 1, 90).reshape([90, 1])

    # print(lin.shape)
    z_ini = np.concatenate([z_ini, zero], axis=1)
    import matplotlib.pyplot as plt
    plt.scatter(z_ini[:,0],z_ini[:,1])
    plt.show()
    # print(z_ini.shape)
    # exit()

    # mmscaler = MinMaxScaler(feature_range=(-0.1, 0.1), copy=True)

    # mmscaler.fit(z_ini)  # xの最大・最小を計算
    # z_nor = mmscaler.transform(z_ini)
    # plt.scatter(z_nor[:, 0], z_nor[:, 1])
    # plt.show()


    # print(z_nor)
    ##############################
    ukr = UKR(X, latent_dim, sigma, prior='random', Zinit=z_ini)
    ukr.fit(epoch, eta, alpha, norm)
    # visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False,filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/UKR動物1")
    # visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/UKR動物1", label=coffee_label)

    #----------描画部分が実装されたらコメントアウト外す----------
    ukr.calc_approximate_f(resolution=15)
    #########観測空間あり#############
    visualize_history_obs(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=True, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/2ZeminewPCA0_0000001", label=label)
    # visualize_PNG_obs(an, X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/UKR2angl-45")

    ############観測空間なし#####################
    # visualize_history_no_obs(X, ukr.history['y'], ukr.history['z'], ukr.history['error'],  save_gif=False, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/2Zemi0PCA3PCAL2", label=label)
    # visualize_PNG_no_obs(an, X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/UKR1顔10")
##-------------画像出力----------#
    # plt.scatter(ukr.history['z'][-1][:,0], ukr.history['z'][-1][:,1])
    # plt.show()
    # img(r, latent_dim)








