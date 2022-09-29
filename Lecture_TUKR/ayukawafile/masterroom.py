import numpy as np
import jax,jaxlib
import jax.numpy as jnp
from tqdm import tqdm #プログレスバーを表示させてくれる
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.decomposition import PCA #主成分分析
# from face_project.load import load_angle_resized_data
# from face_project.load import load_angle_resized_same_angle_data
# from sklearn.manifold import TSNE
# from face_project.load import load_angle_resized_data_TUKR




from Lecture_TUKR.ayukawafile.ayukawa_UKR import UKR
from Lecture_TUKR.ayukawafile.ayukawa_TUKR_animal import TUKR
# from Lecture_TUKR.ayukawafile.animals import load_data
# from Lecture_TUKR.ayukawafile.UKR_vis import visualize_UKRfig_history
# from Lecture_TUKR.ayukawafile.UKR_vis import visualize_UKR_history
# from Lecture_TUKR.ayukawafile.visualizer import visualize_fig_history
# from Lecture_TUKR.ayukawafile.visualizer import visualize_history
from Lecture_TUKR.ayukawafile.SUSHI import load_data
from Lecture_TUKR.ayukawafile.sushi_uke_vis import visualize_UKR_history
from Lecture_TUKR.ayukawafile.sushi_uke_vis import visualize_UKRfig_history
from Lecture_TUKR.ayukawafile.sushi_vis import visualize_fig_history
from Lecture_TUKR.ayukawafile.sushi_vis import visualize_history

# 各種パラメータ変えて遊んでみてね．
##
UKR_epoch = 5000  # 学習回数
TUKR_epoch = 5000
sigma = 0.1  # カーネルの幅
xsigma = 0.0077  # カーネルの幅 フィッティングの強度のイメージ　小さいほどその点が持つ引力？が強くなる
ysigma = 0.1
UKR_eta = 0.08  # 学習率
TUKR_eta = 0.0077
# latent_dim = 1 #潜在空間の次元
latent_dim = 2  # 潜在空間の次元
xlatent_dim = 2  # 潜在空間の次元
ylatent_dim = 2  # 潜在空間の次元  animal
alpha = 0.0001
norm = 10
seed = 2
jedi = 3  # PCAの次元

np.random.seed(seed)
# print(load_data()[0])
X_UKR = load_data()[0]
# print(load_data()[1])
animal = load_data()[1]
pca = PCA(n_components=jedi)  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言 n_componentsで主成分数を定義

df = X_UKR.reshape(X_UKR.shape[0], -1)

pca.fit(df)
kiyo = pca.explained_variance_ratio_
PCA_ans = pca.transform(df)
# #入力データ（詳しくはdata.pyを除いてみると良い）
x = PCA_ans  # 鞍型データ　ob_dim=3, 真のL=2

ukr = UKR(X_UKR, latent_dim, sigma, prior='random')
ukr.fit(UKR_epoch, UKR_eta, alpha, norm)
# print(x.shape)
visualize_UKR_history(X_UKR, ukr.history['kernel'], ukr.history['z'], ukr.history['error'], jedi,  animal, save_gif=False, filename="tmp")


kiyo_goukei = np.add.accumulate(kiyo)

ruisekikiyo = np.hstack([0, kiyo.cumsum()])
print(PCA_ans.shape)
print(kiyo_goukei)
print()
print(ruisekikiyo)


X_TUKR = load_data() #animal
# print(X_TUKR)
# exit()


Tukr = TUKR(X_TUKR, xlatent_dim, ylatent_dim, xsigma, ysigma, prior='random')
Tukr.fit(TUKR_epoch, TUKR_eta, alpha, norm)

Tukr.calc_approximate_f(10, TUKR_epoch)
print(Tukr.history['z'].shape)
# visualize_history(X_TUKR[0][:, :, None], Tukr.history['f'], Tukr.history['z'], Tukr.history['v'], Tukr.history['error'], X_TUKR, save_gif=True, filename="tmp")
visualize_fig_history(X_TUKR[0][:, :, None], Tukr.history['f'], Tukr.history['z'], Tukr.history['v'], Tukr.history['error'], X_TUKR, save_gif=True, filename="tmp")
