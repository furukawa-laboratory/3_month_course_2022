import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm #プログレスバーを表示させてくれる
import jax
import jax.numpy as jnp
from sklearn.decomposition import PCA
from PIL import Image
import os
# from Lecture_UKR.data import create_2d_sin_curve
# from face_project.fukunaga.TUKR_visualizer import visualize_history
from face_project.fukunaga.TUKR_visualizer import visualize_history_obs
from face_project.fukunaga.TUKR_visualizer import visualize_PNG_obs
from face_project.fukunaga.TUKR_visualizer import visualize_PNG_no_obs
from face_project.fukunaga.TUKR_visualizer import visualize_history_no_obs
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from face_project.fukunaga.load import load_angle_resized_data
from sklearn.manifold import TSNE
from face_project.fukunaga.load import load_angle_resized_same_angle_data
from face_project.fukunaga.load import load_angle_resized_data_TUKR
from tensorly.decomposition import tucker
import tensorly as tl




class UKR:
    def __init__(self, X, latent_dim1, latent_dim2, sigma1, sigma2, prior='random', Uinit=None, Vinit=None):
        #--------初期値を設定する．---------
        self.X = X
        #ここから下は書き換えてね
        self.xsamples = X.shape[0]
        self.ysamples = X.shape[1]
        self.ob_dim = X.shape[2]
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2


        if Uinit is None:
            if prior == 'random': #一様事前分布のとき
                self.U = np.random.uniform(low=-0.001, high=0.001, size=(self.xsamples, self.latent_dim1))


            else: #ガウス事前分布のとき
                self.U = np.random.normal(0, self.sigma1, (self.xsamples, self.latent_dim1))

        else: #Zの初期値が与えられた時
            self.U = Uinit
        self.history = {}
        if Vinit is None:
            if prior == 'random': #一様事前分布のとき
                self.V = -np.random.uniform(low=-0.001, high=0.001, size=(self.ysamples, self.latent_dim2))

            else: #ガウス事前分布のとき
                self.V = -np.random.normal(self.ysamples*self.latent_dim2).reshape(self.ysamples, self.latent_dim2)
        else: #Zの初期値が与えられた時
            self.V = Vinit

        self.history = {}
    def f(self, U, V):
        Ud = np.sum((U[:, None, :] - self.U[None, :, :])**2, axis=2)
        UH = -1*(Ud/(2*self.sigma1**2))
        Uh = jnp.exp(UH)

        Vd = np.sum((V[:, None, :] - self.V[None, :, :]) ** 2, axis=2)
        VH = -1 * (Vd / (2 * self.sigma2 ** 2))
        Vh = jnp.exp(VH)

        bunshi = jnp.einsum('KI,MJ,IJD -> KMD', Uh, Vh, self.X)
        # aaa = jnp.einsum('MJ,IJD -> IMD', Vh, self.X)
        # bunshi = jnp.einsum('KI,IMD -> KMD', Uh, aaa)

        bunbo = jnp.einsum('KI,MJ -> KM', Uh, Vh)
        newbunbo = bunbo[:, :, None]



        f = bunshi/newbunbo
        # print(U.shape)
        #写像の計算
        return f
    def ff(self, U, V, epoch):
        Ud = np.sum((U[:, None, :] - self.history['u'][epoch][None, :, :]) ** 2, axis=2)
        UH = -1 * (Ud / (2 * self.sigma1 ** 2))
        Uh = jnp.exp(UH)

        Vd = np.sum((V[:, None, :] - self.history['v'][epoch][None, :, :]) ** 2, axis=2)
        VH = -1 * (Vd / (2 * self.sigma2 ** 2))
        Vh = jnp.exp(VH)

        bunshi = jnp.einsum('KI,MJ,IJD -> KMD', Uh, Vh, self.X)
        # aaa = jnp.einsum('MJ,IJD -> IMD', Vh, self.X)
        # bunshi = jnp.einsum('KI,IMD -> KMD', Uh, aaa)

        bunbo = jnp.einsum('KI,MJ -> KM', Uh, Vh)
        newbunbo = bunbo[:, :, None]

        return bunshi/newbunbo

    def E(self, U, V,  X, alpha, norm): #目的関数の計算
        E = jnp.sum((X - self.f(U, V))**2)
        UR = alpha * jnp.sum(jnp.abs(U ** norm))
        VR = alpha * jnp.sum(jnp.abs(V ** norm))
        E = E/self.xsamples/self.ysamples + UR/self.xsamples + VR/self.ysamples

        return E



    def fit(self, nb_epoch: int, eta: float, alpha: float, norm: float):
        # 学習過程記録用
        self.history['u'] = np.zeros((nb_epoch, self.xsamples, self.latent_dim1))
        self.history['v'] = np.zeros((nb_epoch, self.ysamples, self.latent_dim2))
        self.history['f'] = np.zeros((nb_epoch, self.xsamples, self.ysamples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(np.arange(nb_epoch)):
            dEdu = jax.grad(self.E, argnums=0)(self.U, self.V, self.X, alpha, norm)
            self.U = self.U - eta * dEdu
            dEdv = jax.grad(self.E, argnums=1)(self.U, self.V, self.X, alpha, norm)
            self.V = self.V - eta * dEdv

           # U,Vの更新




            # 学習過程記録用
            self.history['u'][epoch] = self.U
            self.history['v'][epoch] = self.V
            self.history['f'][epoch] = self.f(self.U, self.V)
            self.history['error'][epoch] = self.E(self.U, self.V, self.X, alpha, norm)

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
    def calc_approximate_fu(self, resolutionu, resolutionv): #fのメッシュ描画用，resolution:一辺の代表点の数
        nb_epoch = self.history['u'].shape[0]

        # self.history['y'] = np.zeros((nb_epoch, self.xsamples, self.ysamples, self.ob_dim))
        self.history['y'] = np.zeros((nb_epoch, resolutionu**2, resolutionv, self.ob_dim))
        for epoch in range(nb_epoch):
            uzeta = self.create_uzeta(self.history['u'][epoch], resolutionu)
            vzeta = self.create_vzeta(self.history['v'][epoch], resolutionv)
            Y = self.ff(uzeta, vzeta, epoch)
            # print(self.history['y'][epoch])
            self.history['y'][epoch] = Y
        # print(Y.shape)
        return Y



    def create_uzeta(self, Z, resolutionu): #fのメッシュの描画用に潜在空間に代表点zetaを作る．

        u_y = np.linspace(np.amin(Z), np.amax(Z), self.xsamples).reshape(-1, 1)
        # UYY = np.meshgrid(u_y)
        # A = np.amin(Z)
        # print(self.U.shape)

        # zeta = np.concatenate([uxx[:, None], uyy[:, None]], axis=1)
        z_x = np.linspace(np.min(Z), np.max(Z), resolutionu)
        z_y = np.linspace(np.min(Z), np.max(Z), resolutionu)
        XX, YY = np.meshgrid(z_x, z_y)
        xx = XX.reshape(-1)
        yy = YY.reshape(-1)
        zetau = np.concatenate([xx[:, None], yy[:, None]], axis=1)
        return zetau

    def create_vzeta(self, Z, resolutionv): #fのメッシュの描画用に潜在空間に代表点zetaを作る．

        v_y = np.linspace(np.min(Z), np.max(Z), resolutionv).reshape(-1, 1)
        # UYY = np.meshgrid(u_y)
        # z_x = np.linspace(np.min(Z), np.max(Z), resolution)
        # z_y = np.linspace(np.min(Z), np.max(Z), resolution)
        # XX, YY = np.meshgrid(z_x, z_y)
        # xx = XX.reshape(-1)
        # yy = YY.reshape(-1)
        # zetav = np.concatenate([xx[:, None], yy[:, None]], axis=1)

        # zeta = np.concatenate([uxx[:, None], uyy[:, None]], axis=1)
        return v_y
def img(l, r):
    Y = ukr.calc_approximate_fu(resolutionu=l, resolutionv=r)
    # print(Y.shape)
    Y_inv = pca.inverse_transform(Y)
    # print(Y_inv.shape)
    # print(Y_inv.shape)
    # exit()
    fig = plt.figure(figsize=(8, 8))

    gs = fig.add_gridspec(l, r)
    for i in range(l):
        for j in range(r):
            ax = fig.add_subplot(gs[i, j])
            img = Y_inv[i, j, :]
            img = img.reshape(64, 64)
            plt.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


if __name__ == '__main__':



    ##############################

    #各種パラメータ変えて遊んでみてね．
    epoch = 500 #学習回数
    sigma1 = 0.8 #カーネルの幅
    sigma2 = 0.8
    eta = 0.00001 #学習率
    latent_dim1 = 2 #潜在空間の次元
    latent_dim2 = 1
    alpha = 0.00001
    norm = 10
    seed = 4
    ncom = 100
    np.random.seed(seed)
    l = 10
    r = 13





    x_ori, z1_color, z2_color = load_angle_resized_data_TUKR()
    x = x_ori.reshape(x_ori.shape[0] * x_ori.shape[1], x_ori.shape[2] * x_ori.shape[3])
    # print(x)
    pca = PCA(n_components = ncom)
    # print(78789789789)
    # print(x.shape)
    # print(x.reshape(x.shape[0], -1).shape)
    # print(88888888)

    x_2d = pca.fit_transform(x.reshape(x.shape[0], -1))
    # print(x_2d.shape)

    # print(x_2d.shape)
    # return x_2d
    # return x_2d, z1_color, z2_color
    X, z1_color, z2_color = x_2d,z1_color,z2_color
    cr = pca.explained_variance_ratio_
    # 累積寄与率
    ccr = np.add.accumulate(cr)
    # print(ccr)
    # ---------------　PCA初期化---------------------
    # print(x_ori.shape)
    uu = x_ori[:, 0, :]
    vv = x_ori[0, :, :]
    # print(x_ori.shape)
    uu_reshape = x_ori.reshape(90, -1)
    vv_reshape = x_ori.reshape(33, -1)
    # print(uu_reshape.shape)
    # print(vv_reshape.shape)
    pca_creatu = PCA(n_components=latent_dim1)
    pca_creatv = PCA(n_components=latent_dim2)
    u_ini = pca_creatu.fit_transform(uu_reshape.reshape(uu.shape[0], -1))
    v_ini = pca_creatv.fit_transform(vv_reshape.reshape(vv.shape[0], -1))
#------------タッカー分解--------------
    # x_sec = x_ori.reshape(90, 33, -1)
    # print(x_sec.shape)
    # core, factors = tucker(x_sec, rank=[2, 2, 1])
    # reconstructed_tensor = tl.tucker_to_tensor(core, factors)
    #
    # print(core.shape)
    # print([c.shape for c in factors])

    # print(u_ini.shape)

    # v_inv = pca_creatv.inverse_transform(v_ini)
    # print(v_inv.shape)
    # im = v_inv[0, :]
    # im = im.reshape(64, 64)
    # plt.imshow(im, cmap='gray')
    # plt.show()
    # exit()


    mmscaler = MinMaxScaler(feature_range=(-0.1, 0.1), copy=True)
    # print(u_ini.shape)
    # print(v_ini.shape)

    mmscaler.fit(u_ini)  # xの最大・最小を計算
    u_nor = mmscaler.transform(u_ini)
    mmscaler.fit(v_ini)  # xの最大・最小を計算
    v_nor = mmscaler.transform(v_ini)

    # --------------------------------------------------


    #入力データ（詳しくはdata.pyを除いてみると良い）
    xsamples = 20 #データ数
    ysamples = 10
    # X, z1_color, z2_color = x_PCA()
    X = X.reshape(90, 33, ncom)
    z1 = np.array(z1_color)
    z2 = np.array(z2_color)
    x1, x2 = np.meshgrid(z1, z2, indexing='ij')
    # print(z1.shape)

    truez = np.concatenate((x1[:, :, np.newaxis], x2[:, :, np.newaxis]), axis=2)
    # X = load_kura_tsom(xsamples, ysamples) #鞍型データ　ob_dim=3, 真のL=2
    # X, truez, z1, z2 = load_kura_tsom(xsamples, ysamples, retz=True)
    zzz = [truez, z1, z2]
    # X = load_date()[0][:, :, None]
    # animal_label = load_date(retlabel_animal=True)[1]
    # feature_label = load_date(retlabel_feature=True)[2]

    #X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    # X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1
    ukr = UKR(X, latent_dim1, latent_dim2, sigma1, sigma2, prior='random', Uinit=None, Vinit=None)
    ukr.fit(epoch, eta, alpha, norm)
    # visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")

    #----------描画部分が実装されたらコメントアウト外す----------
    ukr.calc_approximate_fu(resolutionu=l, resolutionv=r)
    # ukr.calc_approximate_fv(resolution=10)
    # visualize_history(X, ukr.history['y'], ukr.history['u'], ukr.history['v'], ukr.history['error'], save_gif=False, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/TUKR動物", label1=animal_label, label2=feature_label)
    # visualize_history(X, ukr.history['y'], ukr.history['u'], ukr.history['v'], ukr.history['error'], save_gif=False,filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/TUKR顔4", zzz=zzz)
    #-----------------観測空間あり---------------#
    # visualize_history_obs(X, ukr.history['y'], ukr.history['u'], ukr.history['v'], ukr.history['error'], save_gif=False,filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/TUKR顔3", zzz=zzz)
    # visualize_PNG_obs(X, ukr.history['y'], ukr.history['u'], ukr.history['v'], ukr.history['error'], save_gif=True,filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/TUKRpca3", zzz=zzz)
    #-----------------観測空間なし---------------#
    visualize_history_no_obs(ukr.history['y'], ukr.history['u'], ukr.history['v'], ukr.history['error'], save_gif=True, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/TUKR-n", zzz=zzz)
    # visualize_PNG_no_obs(ukr.history['y'], ukr.history['u'], ukr.history['v'], ukr.history['error'], save_gif=True,filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/TUKRpca100", zzz=zzz)

#---------画像出力------------
    img(l, r)

