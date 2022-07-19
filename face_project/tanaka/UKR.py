import numpy as np
from tqdm import tqdm #プログレスバーを表示させてくれる
import jax
import jax.numpy as jnp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
                self.Z =np.random.normal(0, self.sigma*0.00001, (self.nb_samples, self.latent_dim))
            else: #ガウス事前分布のとき
                self.Z =np.random.normal(self.nb_samples*self.latent_dim).reshape(self.nb_samples, self.latent_dim)
        else: #Zの初期値が与えられた時
            self.Z = Zinit

        self.history = {}
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
            dEdx = jax.grad(self.E, argnums=0)(self.Z, self.X, alpha, norm)
            self.Z = self.Z -eta * dEdx

           # Zの更新




            # 学習過程記録用
            self.history['z'][epoch] =self.Z
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

if __name__ == '__main__':
    from face_project.tanaka.UKR_visualizer import visualize_history
    # from face_project.tanaka.PCA import x_PCA
    from sklearn.decomposition import PCA
    from face_project.tanaka.load import load_angle_resized_data
    from sklearn.manifold import TSNE
    from face_project.tanaka.load import load_angle_resized_same_angle_data
    from face_project.tanaka.load import load_angle_resized_data_TUKR

    x = load_angle_resized_data('60')
    # x = load_angle_resized_same_angle_data('0')
    # x = load_angle_resized_data_TUKR()
    # print(x)
    pca = PCA(n_components=3)

    x_2d = pca.fit_transform(x.reshape(x.shape[0], -1))
    # print(x_2d)

    X = x_2d
    # 寄与率
    cr = pca.explained_variance_ratio_
    # 累積寄与率
    ccr = np.add.accumulate(cr)

    print(ccr)

    #各種パラメータ変えて遊んでみてね．
    epoch = 500 #学習回数
    sigma = 0.1 #カーネルの幅
    eta = 0.00000005#学習率
    latent_dim = 2 #潜在空間の次元
    alpha = 0.0001
    norm = 10
    seed = 4
    np.random.seed(seed)

    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples = 100 #データ数
    # X = create_kura(nb_samples) #鞍型データ　ob_dim=3, 真のL=2
    # X = x_PCA()
    # X = x_tsne()
    #X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    # X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1
    # X = load_date()[0]
    # animal_label = load_date(retlabel_animal=True)[1]
    # coffee_label = load_date(retlabel_coffee=True)[1]
    # print(load_date(retlabel_animal=True)[1])

    ukr = UKR(X, latent_dim, sigma, prior='random')
    ukr.fit(epoch, eta, alpha, norm)
    # visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False,filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/UKR動物1")
    # visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/UKR動物1", label=coffee_label)

    #----------描画部分が実装されたらコメントアウト外す----------
    ukr.calc_approximate_f(resolution=30)
    visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/UKR顔tsne")

    r = 10
    Y = ukr.calc_approximate_f(resolution=r**2)
    Y_inv = pca.inverse_transform(Y)
    # print(Y_inv.shape)
    fig = plt.figure(figsize=(10, 10), dpi=80)
    gs = fig.add_gridspec(r, r)
    for i in range(r**2):
        fig.add_subplot(gs[i // r, i % r])
        img = Y_inv[i, :]
        img = img.reshape(64, 64)

        plt.imshow(img, cmap='gray')

    plt.show()




