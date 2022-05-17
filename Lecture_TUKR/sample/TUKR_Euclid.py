from tqdm import tqdm
import numpy as np
import sys
sys.path.append('/Users/flab-mac/Co-ocurrence/libs_ando/dataset')
import kura_tsom#データのimport
import tukr_kura_data
import torch
import math
# 描画関連
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class TUKR2_euclidean:
    def __init__(self, X,bandwidth,latent_dim, is_compact,init="random"):

        #行列が与えられるか，テンソルが与えられるかのやつ
        if X.ndim==2:#行列が与えられた場合
            self.N1 = X.shape[0]  # 文書数
            self.N2 = X.shape[1]  # 単語数
            self.observed_dim = 1  # 次元数
            X=X.reshape((self.N1,self.N2,self.observed_dim))
        else:
            self.N1 = X.shape[0]  # 文書数
            self.N2 = X.shape[1]  # 単語数
            self.observed_dim = X.shape[2]  # 次元数

        self.X=torch.tensor(X.copy())

        # 潜在空間の次元
        if isinstance(latent_dim, (tuple, list)):#潜在空間の次元がモードで違う時
            self.latent_dim1 = latent_dim[0]
            self.latent_dim2 = latent_dim[1]
        elif isinstance(latent_dim, int):#潜在空間の次元がモードで同じ時
            self.latent_dim1 = latent_dim
            self.latent_dim2 = latent_dim
        else:
            raise ValueError("invalid latent_dim: {}".format(latent_dim))

        # 潜在空間を区切るか
        self.is_compact = is_compact

        # カーネル幅に関する例外処理
        if isinstance(bandwidth, (tuple, list)):  # 文書と単語でカーネル幅が違う時
            self.bandwidth1 = bandwidth[0]
            self.bandwidth2 = bandwidth[1]
        elif isinstance(bandwidth, float):  # 文書と単語でカーネル幅が同じ時
            self.bandwidth1 = bandwidth
            self.bandwidth2 = bandwidth
        else:
            raise ValueError("invalid kernel_bandwidth: {}".format(bandwidth))

        self.beta1 = 1 / (self.bandwidth1 ** 2)  # 文書潜在空間上のカーネルの精度
        self.beta2 = 1 / (self.bandwidth2 ** 2)  # 単語潜在空間上のカーネルの精度

        if isinstance(init, str) and init == 'random':# 一様分布で初期化
            temp_Z1 = np.random.rand(self.N1, self.latent_dim1)
            self.Z1 = (temp_Z1 - np.mean(temp_Z1)) * self.bandwidth1
            self.Z1=torch.tensor(self.Z1,requires_grad=True,dtype=torch.float64)
            temp_Z2 = np.random.rand(self.N2, self.latent_dim2)
            self.Z2 = (temp_Z2 - np.mean(temp_Z2)) * self.bandwidth2
            self.Z2 = torch.tensor(self.Z2, requires_grad=True, dtype=torch.float64)

        # 初期値を与えるとその値を使う
        elif isinstance(init, (tuple, list)) and len(init) == 2:
            if isinstance(init[0], np.ndarray) and init[0].shape == (self.N1, self.latent_dim1):
                self.Z1 = torch.tensor(init[0].copy(),requires_grad=True,dtype=torch.float64)
            else:
                raise ValueError("invalid inits: {}".format(init))
            if isinstance(init[1], np.ndarray) and init[1].shape == (self.N2, self.latent_dim2):
                self.Z2 = torch.tensor(init[1].copy(),requires_grad=True,dtype=torch.float64)
            else:
                raise ValueError("invalid inits: {}".format(init))
        else:
            raise ValueError("invalid inits: {}".format(init))


        # 学習した値を格納する辞書
        self.history = {}

    def _Delta(self, Z):
        Delta = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]  # N*I*latent_dim

        return Delta

    def _density_Kernel(self, Z, kernel_bandwidth):  # N=I
        Delta = self._Delta(Z)
        Dist = torch.sum(Delta * Delta, dim=2)  # N*I
        latent_dim=Z.shape[1]

        dens_K = math.sqrt((1 / (2 * np.pi * pow(kernel_bandwidth, 2))) ** latent_dim) * torch.exp(
            -Dist / (2 * pow(kernel_bandwidth, 2)))  # division_num*N

        dens_K=dens_K/Z.shape[0]

        return dens_K

    def _smoothing_Kernel(self, Z, beta):  # N=I
        Delta = self._Delta(Z)
        Dist = torch.sum(Delta*Delta, dim=2)  # N*I
        H = torch.exp(-0.5 * beta * Dist)   # 文書潜在空間上の密度カーネルN*I
        H_sum = torch.sum(H, dim=1)[:, np.newaxis]
        R = H / H_sum  # 文書潜在空間上の平滑化カーネルDocument(N)*Document(I)
        #print(H_sum.size())
        return R

    def fit(self, epoch_num, learning_rate, lambda_,fixZ=(False, False)):
        #学習過程を保存する用
        self.history["z1"] = np.zeros((epoch_num, self.N1, self.latent_dim1))
        self.history["z2"] = np.zeros((epoch_num, self.N2, self.latent_dim2))
        self.history["f"] = np.zeros((epoch_num, self.N1, self.N2,self.observed_dim))
        self.history["obj"] = np.zeros((epoch_num))  # 誤差関数の値

        #正則化項について
        if isinstance(lambda_, (tuple, list)):#潜在空間の次元がモードで違う時
            lambda_1 = lambda_[0]
            lambda_2 = lambda_[1]
        elif isinstance(lambda_, float):#潜在空間の次元がモードで同じ時
            lambda_1 = lambda_
            lambda_2 = lambda_
        else:
            raise ValueError("invalid lambda_: {}".format(lambda_))

        # fixZについての例外処理
        if isinstance(fixZ, (tuple, list)) and len(fixZ) == 2:
            if isinstance(fixZ[0], bool):
                pass
            elif isinstance(fixZ[1], bool):
                pass
            else:
                raise ValueError("invalid fixZ: {}".format(fixZ))

        for epoch in tqdm(np.arange(epoch_num)):

            # 平滑化カーネルの作成
            R1 = self._smoothing_Kernel(self.Z1, self.beta1)#N*I
            R2 = self._smoothing_Kernel(self.Z2, self.beta2)#N*I

            #写像の作成
            #self.f=torch.einsum("ni,mj,ijd->nmd",R1,R2,self.X)#N1*I1 N2*I2 I1*J1*D -> N1*N2*D
            u_i=torch.einsum("lj,ijd->ild",R2,self.X)#N1*N2*D
            self.f=torch.einsum("ki,ild->kld",R1,u_i)#N1*N2*D

            #print(self.f.size())

            error_func=1 / (self.N1 * self.N2) * torch.sum((self.X - self.f) ** 2)  # N1*N2*observed_dim
            #print(error_func)
            #print(error_func.size())

            #if self.is_compact is True:

            regularizer1 = torch.sum(self.Z1**10)#NIPSの場合
            regularizer2 = torch.sum(self.Z2**10)


            obj_func=error_func+lambda_1*regularizer1+lambda_2*regularizer2
            #print(obj_func)
            #else:
                # 誤差関数の定義
            #    obj_func = error_func

            obj_func.backward()
            

            with torch.no_grad():#勾配情報の抽出# x.gradの値を保持しない
                dEdZ1 = self.Z1.grad
                dEdZ2 = self.Z2.grad
                #潜在変数の更新
                self.Z1 = self.Z1 - learning_rate * dEdZ1
                self.Z2 = self.Z2 - learning_rate * dEdZ2

            # #有界にする場合
            if self.is_compact is True:
                self.Z1=torch.clamp(self.Z1, -1.0,1.0)
                self.Z2 = torch.clamp(self.Z2, -1.0, 1.0)
            else:
                pass

            #historyに入れる
            self.history["z1"][epoch,:,:]=self.Z1.detach().numpy()
            self.history["z2"][epoch,:,:] = self.Z2.detach().numpy()
            self.history["f"][epoch, :, :,:] = self.f.detach().numpy()
            self.history["obj"][epoch]=obj_func.item()

            self.Z1.requires_grad = True#微分した後はrequired_grad=FalseになるのでTrueにする
            self.Z2.requires_grad = True

        #最後に外から参照しそうな変数をnumpy形式に指定おく
        self.Z1=self.Z1.detach().numpy()
        self.Z2 = self.Z2.detach().numpy()
        self.f=self.f.detach().numpy()
        #print(self.Z1)

def _main():
    # データのimport
    xsamples=20
    ysamples = 25

    bandwidth1=0.1
    bandwidth2 = 0.1
    


    #X = kura_tsom.load_kura_tsom(xsamples=xsamples, ysamples=ysamples,ret_truez=False)
    X = tukr_kura_data.load_kura_tsom(xsamples, ysamples, retz=False)

    #初期値を決める
    np.random.seed(15)
    init_Z1=np.random.normal(0, 1.0, (xsamples, 1)) *bandwidth1 * 0.5

    np.random.seed(6)
    init_Z2 = np.random.normal(0, 1.0, (ysamples, 1)) * bandwidth2 * 0.5

    #学習回数
    epoch_num=1000

    tukr = TUKR2_euclidean(X=X,bandwidth=(bandwidth1,bandwidth2), init=(init_Z1,init_Z2),latent_dim=1, is_compact=True)
    tukr.fit(epoch_num=epoch_num, learning_rate=0.1,lambda_=0.1)

    #描画の描画
    fig = plt.figure()
    ax = Axes3D(fig)

    def plot(i):
        ax.cla()
        ax.scatter(X[:, :, 0], X[:, :, 1], X[:, :, 2])
        ax.plot_wireframe(tukr.history['f'][i, :, :, 0], tukr.history['f'][i, :, :, 1],tukr.history['f'][i, :, :, 2])
        plt.title(' t=' + str(i))

    ani = animation.FuncAnimation(fig, plot, frames=epoch_num, interval=100)
    plt.show()
    ani.save('animation1.gif', writer='pillow')


if __name__ == "__main__":  # このファイルを実行した時のみ
    _main()