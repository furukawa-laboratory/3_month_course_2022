import numpy as np
import matplotlib.pyplot as plt
import jax,jaxlib
import jax.numpy as jnp
import os
class TUKR:
    def __init__(self, X, K_num,latent_dim, resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo,sitaikoto,uk_num,vk_num): # 引数はヒントになるかも！
        #引数を用いたselfの宣言
        self.X = X

        self.X1_num, self.X2_num,self.D = X.shape
        self.X_num=self.X1_num*self.X2_num
        self.nn=int(self.X_num**0.5)
        self.L = latent_dim
        self.K=K_num
        self.KK=int(self.K**0.5)
        self.resolution = resolution

        self.ZN1=np.random.normal(0,0.5,(self.X_num,self.L))
        self.ZN1= np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X1_num,self.L))
        self.ZN2 = np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X2_num, self.L))
        self.uk_num=uk_num
        self.vk_num=vk_num
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


        self.loss_type=['loss','loss_mse','loss_L2'] #合計と正則項と二乗誤差
        self.name = ['loss', 'loss_mse', 'loss_L2', 'y_zn', 'y_zk', 'y_zk_wire', 'zn1','zn2', 'realx']
        # 学習過程記録用 historyの初期化
        self.history = {}
        self.history['zn1'] = np.zeros((nb_epoch, self.X1_num, self.L))
        self.history['zn2'] = np.zeros((nb_epoch, self.X2_num, self.L))
        self.history['y_zn'] = np.zeros((nb_epoch, self.X1_num*self.X2_num,self.D))
        self.history['y_zk'] = np.zeros((nb_epoch, self.uk_num,self.vk_num, self.D))
        self.history['y_zk_wire']=np.zeros((nb_epoch,self.uk_num,self.vk_num,self.D))
        self.history['loss']=np.zeros((self.nb_epoch))
        self.history['loss_mse']=np.zeros((self.nb_epoch))
        self.history['loss_L2'] = np.zeros((self.nb_epoch))
        self.history['realx']=np.zeros((self.X_num,self.D))

        #self.history['zn'][0]=self.ZN
        self.history['realx']=self.X



    def make_zk(self,):
        zn_min=np.zeros((2,self.L))
        zn_max=np.zeros((2,self.L))
        for i in range(2):
            for j in range(self.L):
                if(i==0):
                    zn_min[i][j]= np.min(self.ZN1[:,j])
                    zn_max[i][j]= np.max(self.ZN1[:,j])
                elif(i==1):
                    zn_min[i][j]=np.min(self.ZN2[:,j])
                    zn_max[i][j]= np.max(self.ZN2[:,j])

        zk1 = np.linspace(zn_min[0][0], zn_max[0][0], self.uk_num)
        zk2 = np.linspace(zn_min[1][0], zn_max[1][0], self.vk_num)
        zk1=zk1[:,None]
        zk2 = zk2[:, None]
        # zk1 = zk1.reshape(-1, 1)
        # zk1= np.tile(zk1, self.KK)
        #
        # zk2 = np.tile(zk2, (self.KK, 1))
        # print('zk1,zk2')
        # print(zk1.shape,zk2.shape)
        return zk1,zk2

    def fit(self,):
        #テスト時に入力がznのときのyを求める関数
        def kernel_f(target,type=0,jyun=-1):
            if(type==0):#学習時
                d = jnp.sum((target[:, None, :] - target[None, :, ]) ** 2, axis=2)
                k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)

            elif(type==1):#ワイヤーフレーム時
                # print('fe')
                if(jyun==0):#潜在変数1の方
                    d = np.sum((target[:, None, :] - self.ZN1[None, :, ]) ** 2, axis=2)
                    k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
                    # print(d.shape)
                    # exit()
                elif(jyun==1):#潜在変数2の方
                    d = np.sum((target[:, None, :] - self.ZN2[None, :, ]) ** 2, axis=2)
                    k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
                    # print(d.shape)

            return k

        def karnel_jnp(target1,target2,type=0):
            k1= kernel_f(target1,type,0)
            k2=kernel_f(target2,type,1)

            dotti=2

            if(dotti==1):#だめな方法
                k=jnp.einsum('ik,jl->ij',k1,k2)
                Y = jnp.einsum('ij,lnk->ijk', k, self.X)
                k = k[:, :, None]
            elif(dotti==2):#良い方法
                k=jnp.einsum('ik,jl->ijkl',k1,k2)
                Y=jnp.einsum('ijln,lnk->ijk',k,self.X)
                k=jnp.sum(k,axis=3)
                k=jnp.sum(k,axis=2,keepdims=True)

            YY=Y/k
            return YY

        #学習時の損失を求める関数
        def E_jnp(target1,target2,epoch=-1,kotti=-1):
            YY = karnel_jnp(target1,target2)

            loss = {}
            zn1=jnp.sum(self.ramuda * target1 ** 2, axis=1) / self.L
            zn2 = jnp.sum(self.ramuda * target2 ** 2, axis=1) / self.L


            loss['loss_L2'] = np.sum(zn1)/self.X1_num+np.sum(zn2)/self.X2_num
            loss['loss_mse'] = jnp.sum((self.X - YY) ** 2, axis=2) / self.D
            loss['loss_mse']=jnp.sum(loss['loss_mse'],axis=1)/self.X2_num
            loss['loss_mse']=jnp.sum(loss['loss_mse'])/self.X1_num
            loss['loss'] = loss['loss_mse'] + loss['loss_L2']
            if(epoch==-1):
                return loss['loss']
            else:
                for i in self.loss_type:
                    self.history[i][epoch] = np.sum(loss[i])
        for epoch in range(self.nb_epoch):
            print('epoch='+str(epoch))
            #損失のself.ZNでの微分
            dx1=jax.grad(E_jnp, argnums=0)(self.ZN1,self.ZN2)

            self.ZN1=self.ZN1-self.eta*(dx1)

            dx2 = jax.grad(E_jnp, argnums=1)(self.ZN1,self.ZN2)

            self.ZN2=self.ZN2-self.eta*(dx2)

            #データの保存
            if(epoch%self.meyasu==0):
                #損失の保存
                E_jnp(self.ZN1,self.ZN2,epoch)
                print(self.history['loss'][epoch])
                print(self.history['loss_mse'][epoch])
                print(self.history['loss_L2'][epoch])
                # exit()
                #znを用いて生成したYの保存
                ans=karnel_jnp(self.ZN1,self.ZN2)

                self.history['y_zn'][epoch] =ans.reshape(-1,3)

                #zkを用いて生成したYの保存
                zk1,zk2=self.make_zk()
                #print(zk1.shape,zk2.shape)
                ans2 = karnel_jnp(zk1,zk2,1)
                # print(ans2.shape)
                # exit()
                # resolution = ans2.reshape(self.KK, self.KK, 3)
                # self.history['y_zk'][epoch]=ans2
                #print(ans2.shape)
                #print(self.history['y_zk_wire'][epoch].shape)
                self.history['y_zk_wire'][epoch]=ans2
                #潜在空間ZNの保存
                self.history['zn1'][epoch] = self.ZN1
                self.history['zn2'][epoch] = self.ZN2

        #曲面
        def save_data(name,data,syurui):
            np.save(self.sitaikoto+'/' + str(self.basyo) + '/'+syurui+'/'+name, data)

        data=[self.history[i] for i in self.name]
        for i in range(len(self.name)):
            save_data(self.name[i],data[i],'data')

        print('hai')



