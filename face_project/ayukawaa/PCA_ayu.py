import numpy as np
import sklearn 
from sklearn.decomposition import PCA #主成分分析
from face_project.load import load_angle_resized_data
from face_project.load import load_angle_resized_same_angle_data
from sklearn.manifold import TSNE
from face_project.load import load_angle_resized_data_TUKR

def PCA_1():
    pca = PCA(n_components=30)  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言 n_componentsで主成分数を定義
    x = load_angle_resized_data()
    # x = load_angle_resized_same_angle_data()
    df = x.reshape(x.shape[0], -1)
    # pca.fit(load_angle_resized_data)# PCA を実行
    # PCA_ans = pca.transform(load_angle_resized_data)
    pca.fit(df)
    kiyo = pca.explained_variance_ratio_
    PCA_ans = pca.transform(df)
    print(PCA_ans.shape)
    print(kiyo)
    return PCA_ans

# x = load_angle_resized_data
# print(PCA_ans.shape)

def x_tsne():
    x = load_angle_resized_data()
    tsne = TSNE(n_components=3, random_state=0)
    x_reduce_tsne = tsne.fit_transform(x.reshape(x.shape[0], -1))
    # print(x_reduce_tsne)
    return x_reduce_tsne

#第2主成分まででプロット
# plt.figure(figsize=(6, 6))
# plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df.iloc[:, 0]))
# plt.grid()
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()
#
# #寄与率表示
