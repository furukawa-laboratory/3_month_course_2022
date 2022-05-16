import numpy as np
import matplotlib.pyplot as plt
import jax,jaxlib
import jax.numpy as jnp
import os
class UKR:
    def __init__(self, X, K_num,latent_dim, resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo,sitaikoto): # 引数はヒントになるかも！
        #引数を用いたselfの宣言
        self.X = X
        self.N, self.D = X.shape
        self.nn=int(self.N**0.5)
        self.L = latent_dim
        self.K=K_num
        self.KK=int(self.K**0.5)
        self.resolution = resolution
        self.ZN=np.random.normal(0,0.5,(self.N,self.L))
        self.ZN= np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.N,self.L))
        self.Y = 0
        self.sigma=sigma
        self.ramuda=ramuda
        self.eta=eta
        self.nb_epoch=nb_epoch
        self.meyasu=meyasu
        self.x_range=x_range
        self.sitaikoto=sitaikoto
        self.basyo=basyo

        self.loss_type=['loss','loss_mse','loss_L2'] #合計と正則項と二乗誤差
        self.name = ['loss', 'loss_mse', 'loss_L2', 'y_zn', 'y_zk', 'y_zk_wire', 'zn', 'realx']
        # 学習過程記録用 historyの初期化
        self.history = {}
        self.history['zn'] = np.zeros((nb_epoch, self.N, self.L))
        self.history['y_zn'] = np.zeros((nb_epoch, self.N, self.D))
        self.history['y_zk'] = np.zeros((nb_epoch, self.K, self.D))
        self.history['y_zk_wire']=np.zeros((nb_epoch,self.KK,self.KK,self.D))
        self.history['loss']=np.zeros((self.nb_epoch))
        self.history['loss_mse']=np.zeros((self.nb_epoch))
        self.history['loss_L2'] = np.zeros((self.nb_epoch))
        self.history['realx']=np.zeros((self.N,self.D))
        self.history['zn'][0]=self.ZN
        self.history['realx']=self.X

    def make_zk(self,):
        zn_min = np.min(self.ZN)
        zn_max = np.max(self.ZN)
        zk_x = np.linspace(zn_min, zn_max, self.KK)
        zk_y = np.linspace(zn_min, zn_max, self.KK)
        m_x, m_y = np.meshgrid(zk_x, zk_y)
        m_x = m_x.reshape(-1)
        m_y = m_y.reshape(-1)
        return np.concatenate((m_x[:, None], m_y[:, None]), axis=1)

    def fit(self,):

        #テスト時に入力がznのときのyを求める関数
        def karnel_jnp(target):
            d = np.sum((target[:, None, :] - self.ZN[None, :, ]) ** 2, axis=2)
            k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
            KK=jnp.sum(k,axis=1)
            Y=jnp.einsum('ij,jk->ik',k, self.X)
            YY=Y/KK[:,None]
            return YY

        #学習時の損失を求める関数
        def E_jnp(target,epoch=-1):
            YY = karnel_jnp(target)
            loss = {}
            loss['loss_L2'] = jnp.sum(self.ramuda * target ** 2, axis=1) / 2 / self.N
            loss['loss_mse'] = jnp.sum((self.X - YY) ** 2, axis=1) / 2 / self.N
            loss['loss'] = jnp.sum(loss['loss_mse'] + loss['loss_L2'])

            if(epoch==-1):
                return loss['loss']
            else:
                for i in self.loss_type:
                    self.history[i][epoch] = jnp.sum(loss[i]) / 2 / self.N

        for epoch in range(self.nb_epoch):
            print('epoch='+str(epoch))
            #損失のself.ZNでの微分1
            dx=jax.grad(E_jnp, argnums=0)(self.ZN)
            self.ZN=self.ZN-self.eta*(dx)
            #データの保存
            if(epoch%self.meyasu==0):
                #損失の保存
                E_jnp(self.ZN,epoch)
                #znを用いて生成したYの保存
                ans=karnel_jnp(self.ZN)
                self.history['y_zn'][epoch] =ans
                #zkを用いて生成したYの保存
                zk=self.make_zk()
                ans2 = karnel_jnp(zk)
                resolution = ans2.reshape(self.KK, self.KK, 3)
                self.history['y_zk'][epoch]=ans2
                self.history['y_zk_wire'][epoch]=resolution
                #潜在空間ZNの保存
                self.history['zn'][epoch] = self.ZN

        #曲面
        def save_data(name,data,syurui):
            np.save(self.sitaikoto+'/' + str(self.basyo) + '/'+syurui+'/'+name, data)

        data=[self.history[i] for i in self.name]
        for i in range(len(self.name)):
            save_data(self.name[i],data[i],'data')

        print('hai')


