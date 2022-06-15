import numpy as np
from sklearn.decomposition import PCA
from face_project.load import load_angle_resized_data
from face_project.load import load_angle_resized_same_angle_data
from sklearn.manifold import TSNE
from face_project.load import load_angle_resized_data_TUKR
import matplotlib.pyplot as plt
def x_PCA():
    # x = load_angle_resized_data()
    #x = load_angle_resized_same_angle_data()
    x = load_angle_resized_data_TUKR()
    # print(x)
    pca = PCA(n_components=3)
    x_2d = pca.fit_transform(x.reshape(x.shape[0], -1))
    # print(x_2d)
    return x_2d
def x_tsne():
    x = load_angle_resized_data()
    tsne = TSNE(n_components=3, random_state=0)
    x_reduce_tsne = tsne.fit_transform(x.reshape(x.shape[0], -1))
    # print(x_reduce_tsne)
    return x_reduce_tsne

# print(x_2d)
# # 寄与率
# cr = pca.explained_variance_ratio_
# # 累積寄与率
# ccr = np.add.accumulate(cr)
# # 固有値
# eigenvalue = pca.explained_variance_
# # 固有ベクトル
# eigenvector = pca.components_
# print(ccr)
# fig, ax = plt.subplots()
# ax.plot(ccr)
# ax.set_ylabel('CCR', fontsize=15)
#plt.show()
# from scipy.linalg import svd
# from sklearn.decomposition import TruncatedSVD
# np.set_printoptions(suppress=True)
# U, s, VT = svd(x.reshape(x.shape[0], -1))
# # print("This is U: " + str(U))
# # print("This is s: " + str(s))
# # print("This is V^T: " + str(VT))
# svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
# svd.fit(x.reshape(x.shape[0], -1))
#
# X = svd.transform(x.reshape(x.shape[0], -1))
#
# print("This is matrix after dimentionality reduction: " + str(X))
#