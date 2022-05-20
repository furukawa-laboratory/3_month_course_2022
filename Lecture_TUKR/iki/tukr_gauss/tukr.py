import numpy as np
import matplotlib.pyplot as plt
import jax,jaxlib
import jax.numpy as jnp
import os
class TUKR:
    def __init__(self, X, K_num,latent_dim, resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo,sitaikoto,X_graph_mode,X1_num,X2_num,X3_num): # 引数はヒントになるかも！
        #引数を用いたselfの宣言
        self.X = X_graph_modefff
        self.X_graph_mode=X_graph_mode #50,3

        self.ZN1_num,self.ZN2_num, self.D = X.shape
        self.nn=int(self.ZN1_num*self.ZN2_num**0.5)
        self.L = latent_dim
        self.K=K_num
        self.KK=int(self.K**0.5)
        self.resolution = resolution

        self.ZN1=np.random.normal(0,0.5,(self.ZN1_num,self.L))
        self.ZN1= np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.ZN1_num,self.L))
        self.ZN2 = np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.ZN2_num, self.L))

        # self.

        self.Y = 0
        self.sigma=sigma
        self.ramuda=ramuda
        self.eta=eta
        self.nb_epoch=nb_epoch
        self.meyasu=meyasu
        self.x_range=x_range
        self.sitaikoto=sitaikoto
        self.basyo=basyo

        #tukr
        self.X1_num=X1_num
        self.X2_num = X2_num
        self.X3_num = X3_num
        self.Z1_num=X1_num
        self.Z2_num = X2_num
        self.Z3_num = X3_num

        self.loss_type=['loss','loss_mse','loss_L2'] #合計と正則項と二乗誤差
        self.name = ['loss', 'loss_mse', 'loss_L2', 'y_zn', 'y_zk', 'y_zk_wire', 'zn', 'realx']
        # 学習過程記録用 historyの初期化
        self.history = {}
        self.history['zn1'] = np.zeros((nb_epoch, self.ZN1_num, self.L))
        self.history['zn2'] = np.zeros((nb_epoch, self.ZN2_num, self.L))
        self.history['y_zn'] = np.zeros((nb_epoch, self.ZN1_num*self.ZN2_num, self.D))
        # self.history['y_zk'] = np.zeros((nb_epoch, self.K, self.D))
        # self.history['y_zk_wire']=np.zeros((nb_epoch,self.KK,self.KK,self.D))
        self.history['loss']=np.zeros((self.nb_epoch))
        self.history['loss_mse']=np.zeros((self.nb_epoch))
        self.history['loss_L2'] = np.zeros((self.nb_epoch))
        self.history['realx']=np.zeros((self.ZN1_num*self.ZN2_num,self.D))
        self.history['X_graph_']=np.zeros((self.X1_num*self.X2_num,3))

        #self.history['zn'][0]=self.ZN
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
        def kernel_f(target):
            d = np.sum((target[:, None, :] - target[None, :, ]) ** 2, axis=2)
            k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
            return k

        def karnel_jnp(target1,target2):
            # d = np.sum((target[:, None, :] - self.ZN[None, :, ]) ** 2, axis=2)
            # k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
            k1= kernel_f(target1,)
            k2=kernel_f(target2,)
            k=jnp.einsum('ik,jl->ij',k1,k2)
            # print(k.shape)
            # exit()
            # print(k1.shape,k2.shape)
            # k1=k1[:,None,:,None]
            # k2=k2[None,:,None,:]
            # print(k1.shape, k2.shape)
            #
            # k=jnp.einsum('ikil,ljkj->ij',k1,k2)
            # print(k.shape)

            #k=jnp.einsum('')
            # print(k.shape,self.X.shape)

            Y=jnp.einsum('ij,ijk->ijk',k,self.X)
            # print(Y.shape,k.shape,k[:,:,None].shape)
            YY=Y/k[:,:,None]
            # print(YY.shape)

            return YY

        #学習時の損失を求める関数
        def E_jnp(target1,target2,epoch=-1):
            YY = karnel_jnp(target1,target2)

            loss = {}
            # print(target1.shape)
            a=jnp.sum(self.ramuda * target1 ** 2, axis=1) / self.L
            # print(a.shape)
            # print(jnp.sum(a).shape)
            # print(jnp.sum(a))
            # print((jnp.sum(jnp.sum(self.ramuda * target1 ** 2, axis=1) / 2 / self.ZN1_num)).shape)
            # print((jnp.sum(self.ramuda * target2 ** 2, axis=1) / 2 / self.ZN2_num).shape)

            zn1=jnp.sum(self.ramuda * target1 ** 2, axis=1) / self.L
            zn1=jnp.sum(zn1)/self.ZN1_num
            zn2 = jnp.sum(self.ramuda * target2 ** 2, axis=1) / self.L
            zn2 = jnp.sum(zn2) / self.ZN2_num
            # print(zn1,zn2)
            # print(self.X.shape,YY.shape)
            loss['loss_L2'] = zn1+zn2

            loss['loss_mse'] = jnp.sum((self.X - YY) ** 2, axis=2) / 2 / (self.ZN1_num*self.ZN2_num)

            loss['loss_mse']=jnp.sum(loss['loss_mse'],axis=1)/self.ZN2_num

            loss['loss_mse'] = jnp.sum(loss['loss_mse'])/self.ZN1_num

            loss['loss'] = jnp.sum(loss['loss_mse'] + loss['loss_L2'])
            # print('jijo')
            # print(loss['loss_L2'].shape)
            # print(loss['loss_mse'].shape)
            # exit()
            if(epoch==-1):
                return loss['loss']
            else:
                # print(loss['loss'].shape)
                # print(loss['loss_mse'].shape)
                # print(loss['loss_L2'].shape)
                for i in self.loss_type:
                    # print(i)
                    # print(self.history[i][epoch])
                    # print(loss[i])
                    # print(loss[i].shape)
                    self.history[i][epoch] = loss[i]
        for epoch in range(self.nb_epoch):
            print('epoch='+str(epoch))
            #損失のself.ZNでの微分1

            dx1=jax.grad(E_jnp, argnums=0)(self.ZN1,self.ZN2)
            self.ZN1=self.ZN1-self.eta*(dx1)
            dx2 = jax.grad(E_jnp, argnums=1)(self.ZN1,self.ZN2)
            self.ZN2=self.ZN2-self.eta*(dx2)
            # print(dx1,dx2)
            #データの保存
            if(epoch%self.meyasu==0):
                #損失の保存
                E_jnp(self.ZN1,self.ZN2,epoch)
                print(self.history['loss'][epoch])
                # exit()
                #znを用いて生成したYの保存
                ans=karnel_jnp(self.ZN1,self.ZN2)
                print(ans.shape)
                print( self.history['y_zn'].shape)
                exit()
                self.history['y_zn'][epoch] =ans
                # #zkを用いて生成したYの保存
                # zk=self.make_zk()
                # ans2 = karnel_jnp(zk)
                # resolution = ans2.reshape(self.KK, self.KK, 3)
                # self.history['y_zk'][epoch]=ans2
                # self.history['y_zk_wire'][epoch]=resolution
                # #潜在空間ZNの保存
                # self.history['zn'][epoch] = self.ZN

        #曲面
        def save_data(name,data,syurui):
            np.save(self.sitaikoto+'/' + str(self.basyo) + '/'+syurui+'/'+name, data)
        exit()
        data=[self.history[i] for i in self.name]
        for i in range(len(self.name)):
            save_data(self.name[i],data[i],'data')

        print('hai')


