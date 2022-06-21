import numpy as np
import sklearn 
from sklearn.decomposition import PCA #主成分分析
from face_project.load import load_angle_resized_data

def PCA_1():
    pca = PCA(n_components=3)  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言 n_componentsで主成分数を定義
    x = load_angle_resized_data()
    df = x.reshape(x.shape[0], -1)
    # pca.fit(load_angle_resized_data)# PCA を実行
    # PCA_ans = pca.transform(load_angle_resized_data)
    pca.fit(df)
    PCA_ans = pca.transform(df)
    print(PCA_ans.shape)
    # print(x.shape)
    return PCA_ans

# x = load_angle_resized_data

# print(PCA_ans.shape)


#第2主成分まででプロット
# plt.figure(figsize=(6, 6))
# plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df.iloc[:, 0]))
# plt.grid()
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()
#
# #寄与率表示
# pca.explained_variance_ratio_