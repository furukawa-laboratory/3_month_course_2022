import numpy as np
import matplotlib.pyplot as plt
import jax,jaxlib
import jax.numpy as jnp
import os
class UKR:
    def __init__(self, X, K_num,latent_dim, resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo,zk_omomi): # 引数はヒントになるかも！
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
        self.zk_omomi=zk_omomi
        # 学習過程記録用
        self.history = {}
        self.history['zn'] = np.zeros((nb_epoch, self.N, self.L))
        self.history['y_zn'] = np.zeros((nb_epoch, self.N, self.D))
        self.history['y_zk'] = np.zeros((nb_epoch, self.K, self.D))
        self.history['y_zk_wire']=np.zeros((nb_epoch,self.KK,self.KK,self.D))
        self.history['e']=np.zeros((self.nb_epoch))
        self.history['e_seisoku']=np.zeros((self.nb_epoch))
        self.history['e_loss'] = np.zeros((self.nb_epoch))
        self.e=np.zeros(1)
        self.e_seisoku = np.zeros(1)
        self.e_loss = np.zeros(1)

        zn_min = np.min(self.ZN)*self.zk_omomi
        zn_max = np.max(self.ZN)*self.zk_omomi
        zk_x = np.linspace(zn_min, zn_max, self.KK)
        zk_y = np.linspace(zn_min, zn_max, self.KK)
        m_x, m_y = np.meshgrid(zk_x, zk_y)
        m_x = m_x.reshape(-1)
        m_y = m_y.reshape(-1)
        self.zk = np.concatenate((m_x[:, None], m_y[:, None]), axis=1)
        self.history['zn'][0]=self.ZN
        self.basyo=basyo


    def fit(self,):

        #テスト時に入力がznのときのyを求める関数
        def karnel(ZN,sigma,X):
            d = np.sum((ZN[:, None, :] - ZN[None, :, ]) ** 2, axis=2)
            k = jnp.exp(-1 / (2 * sigma ** 2) * d)
            KK=jnp.sum(k,axis=1)
            Y=np.einsum('ij,jk->ik',k, X)
            YY=Y/KK[:,None]
            return YY

        #テスト時に入力がzkのときのyを求める関数
        def karnel2(zz,ZN,sigma,X):
            d = np.sum((zz[:, None, :] - ZN[None, :, ]) ** 2, axis=2)
            k = np.exp(-1 / (2 * sigma ** 2) * d)
            KK=np.sum(k,axis=1)
            Y=np.einsum('ij,jk->ik',k, X)
            YY=Y/KK[:,None]
            return YY

        #学習時の損失を求める関数
        def E(ZN, sigma, X,ramuda,epoch):
            #カーネル平滑化
            d = np.sum((ZN[:, None, :] - ZN[None, :, ]) ** 2, axis=2)
            k = jnp.exp(-1 / (2 * sigma ** 2) * d)
            KK=jnp.sum(k,axis=1)
            Y= jnp.einsum('ij,jk->ik',k, X)
            YY=Y/KK[:,None]
            #正則化項を加えた損失計算

            seisoku=jnp.sum(ramuda*ZN**10,axis=1)
            seisoku=seisoku/self.N

            loss=jnp.sum((X-YY)**2,axis=1)
            loss=loss/2/self.N
            e=jnp.sum(loss+seisoku)

            return e

        def E_hozon(ZN, sigma, X,ramuda,epoch):
            # print(567676,epoch)
            #カーネル平滑化
            d = np.sum((ZN[:, None, :] - ZN[None, :, ]) ** 2, axis=2)
            k = np.exp(-1 / (2 * sigma ** 2) * d)
            KK=np.sum(k,axis=1)
            Y= np.einsum('ij,jk->ik',k, X)
            YY=Y/KK[:,None]
            #正則化項を加えた損失計算

            seisoku=np.sum(ramuda*ZN**10,axis=1)
            seisoku=seisoku/self.N
            seisoku1=np.sum(seisoku)/self.N
            loss=np.sum((X-YY)**2,axis=1)
            loss=loss/2/self.N
            loss1=np.sum(loss)/2/self.N
            e=np.sum(loss+seisoku)
            self.history['e_seisoku'][epoch]=seisoku1
            self.history['e_loss'][epoch] = loss1

            self.history['e'][epoch] = e

            return e

        for epoch in range(self.nb_epoch):
            print('epoch='+str(epoch))
            #損失のself.ZNでの微分
            dx=jax.grad(E, argnums=0)(self.ZN,self.sigma,self.X,self.ramuda,epoch)
            E_hozon(self.ZN,self.sigma,self.X,self.ramuda,epoch)
            self.ZN=self.ZN-self.eta*(dx)
            #画像と配列の保存
            if(epoch%self.meyasu==0):
                #znを用いて生成したYのscatter画像
                ans=karnel(self.ZN,self.sigma,self.X)

                self.history['y_zn'][epoch] =ans
                #zkを用いて生成したYのワイヤー画像
                zn_min=np.min(self.ZN)*self.zk_omomi
                zn_max=np.max(self.ZN)*self.zk_omomi
                zk_x = np.linspace(zn_min, zn_max, self.KK)
                zk_y = np.linspace(zn_min, zn_max, self.KK)
                m_x, m_y = np.meshgrid(zk_x, zk_y)
                m_x = m_x.reshape(-1)
                m_y = m_y.reshape(-1)
                self.zk = np.concatenate((m_x[:, None], m_y[:, None]), axis=1)
                ans2 = karnel2(self.zk,self.ZN, self.sigma, self.X)
                resolution = ans2.reshape(self.KK, self.KK, 3)
                self.history['y_zk'][epoch]=ans2
                self.history['y_zk_wire'][epoch]=resolution

                if(epoch!=0):
                    self.history['zn'][epoch] = self.ZN


        #曲面
        print(np.max(self.ZN))
        print(np.average(self.ZN))
        print(self.history['e'].shape,self.history['e_seisoku'].shape,self.history['e_loss'].shape)
        np.save('data/' + str(self.nb_epoch) + '/'+str(self.basyo)+'/realx_'+str(0) , self.X)
        np.save('data/' + str(self.nb_epoch) + '/' + str(self.basyo) + '/e' , self.history['e'])
        np.save('data/' + str(self.nb_epoch) + '/' + str(self.basyo) + '/e_seisoku', self.history['e_seisoku'])
        np.save('data/' + str(self.nb_epoch) + '/' + str(self.basyo) + '/e_loss', self.history['e_loss'])
        np.save('data/' + str(self.nb_epoch) + '/' + str(self.basyo) + '/y_zn' , self.history['y_zn'])
        np.save('data/' + str(self.nb_epoch) + '/' + str(self.basyo) + '/y_zk', self.history['y_zk'])
        np.save('data/' + str(self.nb_epoch) + '/' + str(self.basyo) + '/y_zk_wire', self.history['y_zk_wire'])
        np.save('data/' + str(self.nb_epoch) + '/' + str(self.basyo) + '/zn', self.history['zn'])



