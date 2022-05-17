import numpy as np
from tqdm import tqdm #プログレスバーを表示させてくれる


class UKR:
    def __init__(self, X, latent_dim, sigma, prior='random', Zinit=None):
        #--------初期値を設定する．---------
        self.X = X
        #ここから下は書き換えてね
        self.nb_samples, self.ob_dim =
        self.sigma =
        self.latent_dim =

        if Zinit is None:
            if prior == 'random': #一様事前分布のとき
                Z =
            else: #ガウス事前分布のとき
                Z =
        else: #Zの初期値が与えられた時
            Z = Zinit

        self.history = {}

    def f(self, Z1, Z2): #写像の計算





        return f

    def E(self, Z, X, alpha=0, norm=2): #目的関数の計算



        return E

    def fit(self, nb_epoch: int, eta: float) :
        # 学習過程記録用
        self.history['z'] = np.zeros((nb_epoch, self.nb_samples, self.latent_dim))
        self.history['f'] = np.zeros((nb_epoch, self.nb_samples, self.ob_dim))
        self.history['error'] = np.zeros(nb_epoch)

        for epoch in tqdm(range(nb_epoch)):
           # Zの更新




            # 学習過程記録用
            self.history['z'][epoch] =
            self.history['f'] =
            self.history['error'][epoch] =

    #--------------以下描画用(上の部分が実装できたら実装してね)---------------------
#     def calc_approximate_f(self, resolution): #fのメッシュ描画用，resolution:一辺の代表点の数
#         nb_epoch = self.history['z'].shape[0]
#         self.history['y'] = np.zeros((nb_epoch, resolution ** self.latent_dim, self.ob_dim))
#         for epoch in tqdm(range(nb_epoch)):
#
#
#
#             self.history['y'][epoch] = Y
#         return self.history['y']
#
#
# def create_zeta(Z, resolution): #fのメッシュの描画用に潜在空間に代表点zetaを作る．
#
#
#
#
#
#
#     return zeta


if __name__ == '__main__':
    from Lecture_UKR.data import create_kura
    from Lecture_UKR.data import create_rasen
    from Lecture_UKR.data import create_2d_sin_curve
    from visualizer import visualize_history

    #各種パラメータ変えて遊んでみてね．
    epoch = 200 #学習回数
    sigma = 0.1 #カーネルの幅
    eta = 100 #学習率
    latent_dim = 2 #潜在空間の次元

    seed = 4
    np.random.seed(seed)

    #入力データ（詳しくはdata.pyを除いてみると良い）
    nb_samples = 100 #データ数
    X = create_kura(nb_samples) #鞍型データ　ob_dim=3, 真のL=2
    # X = create_rasen(nb_samples) #らせん型データ　ob_dim=3, 真のL=1
    # X = create_2d_sin_curve(nb_samples) #sin型データ　ob_dim=2, 真のL=1

    ukr = UKR(X, latent_dim, sigma, prior='random')
    ukr.fit(epoch, eta)
    visualize_history(X, ukr.history['f'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")

    #----------描画部分が実装されたらコメントアウト外す----------
    # ukr.calc_approximate_f(resolution=10)
    # visualize_history(X, ukr.history['y'], ukr.history['z'], ukr.history['error'], save_gif=False, filename="tmp")



