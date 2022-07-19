import numpy as np

from tqdm import tqdm #プログレスバーを表示させてくれる
import jax
import jax.numpy as jnp


class UKR:
    def __init__(self, X, latent_dim1,latent_dim2, sigma1, sigma2, prior='random', Uinit=None, Vinit=None):
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
                self.V = np.random.uniform(low=-0.001, high=0.001, size=(self.ysamples, self.latent_dim2))

            else: #ガウス事前分布のとき
                self.V = np.random.normal(self.ysamples*self.latent_dim2).reshape(self.ysamples, self.latent_dim2)
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
    def calc_approximate_fu(self, resolution_u, resolution_v): #fのメッシュ描画用，resolution:一辺の代表点の数
        nb_epoch = self.history['u'].shape[0]

        self.history['y'] = np.zeros((nb_epoch, self.xsamples, self.ysamples, self.ob_dim))
        for epoch in range(nb_epoch):
            uzeta = self.create_uzeta(self.history['u'][epoch])
            vzeta = self.create_vzeta(self.history['v'][epoch])
            Y = self.ff(uzeta, vzeta, epoch)
            # print(self.history['y'][epoch])
            self.history['y'][epoch] = Y
        return Y



    def create_uzeta(self, Z, resolution_u): #fのメッシュの描画用に潜在空間に代表点zetaを作る．

        u_y = np.linspace(np.amin(Z), np.amax(Z), self.xsamples).reshape(-1, 1)
        # UYY = np.meshgrid(u_y)
        # A = np.amin(Z)
        # print(self.U.shape)

        # zeta = np.concatenate([uxx[:, None], uyy[:, None]], axis=1)
        z_x = np.linspace(np.min(Z), np.max(Z), resolution_u)
        z_y = np.linspace(np.min(Z), np.max(Z), resolution_u)
        XX, YY =np.meshgrid(z_x,z_y)
        xx = XX.reshape(-1)
        yy = YY.reshape(-1)
        zetau =np.concatenate([xx[:,None],yy[:,None]], axis=1)
        return zetau


    def create_vzeta(self, Z, resolution_v): #fのメッシュの描画用に潜在空間に代表点zetaを作る．

        v_y = np.linspace(np.min(Z), np.max(Z), self.ysamples).reshape(-1, 1)
        # UYY = np.meshgrid(u_y)
        # z_x = np.linspace(np.min(Z), np.max(Z), resolution_v)
        # z_y = np.linspace(np.min(Z), np.max(Z), resolution_v)
        # XX, YY = np.meshgrid(z_x, z_y)
        # xx = XX.reshape(-1)
        # yy = YY.reshape(-1)
        # zetav = np.concatenate([xx[:, None], yy[:, None]], axis=1)
        # return zetav
        return v_y



if __name__ == '__main__':
    # from Lecture_UKR.data import create_2d_sin_curve
    from face_project.fukunaga.TUKR_visualizer import visualize_history
    # from face_project.fukunaga.PCA import x_PCA

    from sklearn.decomposition import PCA
    from face_project.tanaka.load import load_angle_resized_data
    from sklearn.manifold import TSNE
    # from face_project.tanaka.load import load_angle_resized_same_angle_data
    # from face_project.tanaka.load import load_angle_resized_data_TUKR

    # x = load_angle_resized_data('90')
    # x = load_angle_resized_same_angle_data('0')


    #各種パラメータ変えて遊んでみてね．
    epoch = 100 #学習回数
    sigma1 = 0.2 #カーネルの幅
    sigma2 = 0.3
    eta = 5 #学習率
    latent_dim = 1 #潜在空間の次元
    alpha = 0.01
    norm = 2
    seed = 4
    np.random.seed(seed)

    #入力データ（詳しくはdata.pyを除いてみると良い）
    xsamples = 20 #データ数
    ysamples = 10
    X = x_PCA()
    # X = load_kura_tsom(xsamples, ysamples) #鞍型データ　ob_dim=3, 真のL=2
    # X, truez, z1, z2 = load_kura_tsom(xsamples, ysamples, retz=True)
    # zzz = [truez, z1, z2]
    # X = load_date()[0][:, :, None]
    # animal_label = load_date(retlabel_animal=True)[1]
    # feature_label = load_date(retlabel_feature=True)[2]

    #X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    # X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1
    ukr = UKR(X, latent_dim, sigma1, sigma2, prior='random')
    ukr.fit(epoch, eta, alpha, norm)
    #visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")

    #----------描画部分が実装されたらコメントアウト外す----------
    ukr.calc_approximate_fu(resolution=10)
    # ukr.calc_approximate_fv(resolution=10)
    # visualize_history(X, ukr.history['y'], ukr.history['u'], ukr.history['v'], ukr.history['error'], save_gif=False, filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/TUKR動物", label1=animal_label, label2=feature_label)
    visualize_history(X, ukr.history['y'], ukr.history['u'], ukr.history['v'], ukr.history['error'], save_gif=False,filename="/Users/furukawashuushi/Desktop/3ヶ月コースGIF/TUKR", zzz=zzz)

    Y = ukr.calc_approximate_fu(resolution=30,resolution=33)
    r = 10
    Y = ukr.calc_approximate_f(resolution=r ** 2)
    Y_inv = pca.inverse_transform(Y)
    # print(Y_inv.shape)
    fig = plt.figure(figsize=(10, 10), dpi=80)
    gs = fig.add_gridspec(r, r)
    for i in range(r ** 2):
        fig.add_subplot(gs[i // r, i % r])
        img = Y_inv[i, :]
        img = img.reshape(64, 64)

        plt.imshow(img, cmap='gray')

    plt.show()