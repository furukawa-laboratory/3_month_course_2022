import numpy as np
import matplotlib.pyplot as plt
import jax,jaxlib
import jax.numpy as jnp
from jax import jit
import os
class TUKR:
    def __init__(self, X, K_num,latent_dim, resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo,sitaikoto,X1_num,X2_num,X1,X2): # 引数はヒントになるかも！
        #引数を用いたselfの宣言
        self.X = X

        self.X_num, self.D = X.shape
        self.nn=int(self.X_num**0.5)
        self.L = latent_dim
        self.K=K_num
        self.KK=int(self.K**0.5)
        self.resolution = resolution
        self.X1_num=X1_num
        self.X2_num=X2_num

        self.ZN1=np.random.normal(0,0.5,(self.X_num,self.L))
        self.ZN1= np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X_num,self.L))
        self.ZN2 = np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X_num, self.L))
        #record用
        self.ZN1=np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X1_num,self.L))
        self.ZN2 = np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X2_num, self.L))

        #self.ZN=np.stack([self.ZN1,self.ZN2],0)
        # self.ZN=np.array([self.ZN1,self.ZN2])
        # print('fef')
        # print(self.ZN1.shape,self.ZN2.shape,self.ZN.shape)
        # exit()
        self.ZN1_calc=np.zeros((self.X1_num*self.X2_num,self.L))
        self.ZN2_calc = np.zeros((self.X1_num * self.X2_num, self.L))

        # self.ZN1=np.tile(np.random.uniform(low=-self.resolution,high=self.resolution,size=(self.X1_num)),self.X2_num)
        # self.ZN2=
        z1 = np.arange(self.X1_num)
        z2 = np.arange(self.X2_num)
        m_x, m_y = np.meshgrid(z1, z2)
        m_x = m_x.reshape(-1)
        m_y = m_y.reshape(-1)
        self.ZN_nums= np.concatenate((m_x[:, None], m_y[:, None]), axis=1)




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
        self.name = ['loss', 'loss_mse', 'loss_L2', 'y_zn', 'y_zk', 'y_zk_wire', 'zn1','zn2', 'realx','realx1','realx2']
        # 学習過程記録用 historyの初期化
        self.history = {}
        self.history['zn1'] = np.zeros((nb_epoch, self.X1_num, self.L))
        self.history['zn2'] = np.zeros((nb_epoch, self.X2_num, self.L))
        self.history['y_zn'] = np.zeros((nb_epoch, self.X_num, self.D))
        self.history['y_zk'] = np.zeros((nb_epoch, self.K, self.D))
        self.history['y_zk_wire']=np.zeros((nb_epoch,self.KK,self.KK,self.D))
        self.history['loss']=np.zeros((self.nb_epoch))
        self.history['loss_mse']=np.zeros((self.nb_epoch))
        self.history['loss_L2'] = np.zeros((self.nb_epoch))
        self.history['realx']=np.zeros((self.X_num,self.D))
        self.history['realx1'] = np.zeros((self.X1_num, self.D))
        self.history['realx2'] = np.zeros((self.X2_num, self.D))

        #self.history['zn'][0]=self.ZN
        self.history['realx']=self.X
        self.history['realx1'] = X1
        self.history['realx2']=X2


    def make_zk(self,):
        zn_min=np.zeros((2,2))
        zn_max=np.zeros((2,2))
        for i in range(2):
            for j in range(self.L):
                if(i==0):
                    #print(self.ZN1.shape,j)
                    zn_min[i][j]= np.min(self.ZN1[:,j])
                    zn_max[i][j]= np.max(self.ZN1[:,j])
                elif(i==1):
                    zn_min[i][j]= np.min(self.ZN2[:,j])
                    zn_max[i][j]= np.max(self.ZN2[:,j])

            zk_x = np.linspace(zn_min[i][0], zn_max[i][0], self.KK)
            zk_y = np.linspace(zn_min[i][1], zn_max[i][1], self.KK)
            m_x, m_y = np.meshgrid(zk_x, zk_y)
            m_x = m_x.reshape(-1)
            m_y = m_y.reshape(-1)
            if(i==0):
                z1=np.concatenate((m_x[:, None], m_y[:, None]), axis=1)
                # print(z1.shape)
            elif(i==1):
                z2 = np.concatenate((m_x[:, None], m_y[:, None]), axis=1)
        return z1,z2

    def fit(self,):
        #テスト時に入力がznのときのyを求める関数

        def kernel_f(target,type=0,jyun=-1,):
            if(type==0):#学習時
                # print(self.ZN1_calc.shape,self.ZN1.shape,self.ZN_nums.shape)
                # #exit()
                # print(self.ZN1)
                # print(self.ZN1[0])
                # print(self.ZN1[[0,]])
                # print(self.ZN1[0,0])
                # # print(self.ZN_nums)
                # print(self.ZN_nums.shape)
                # print(self.ZN_nums)
                # print(self.ZN_nums[:,0])
                # print(self.ZN1[[self.ZN_nums[:,0]]])
                # print(self.ZN2[self.ZN_nums[:,1]])
                a=target[self.ZN_nums[:,jyun]]
                #self.ZN1_calc=self.ZN1[self.ZN_nums[:,0]]
                # self.ZN2_calc=self.ZN2[self.ZN_nums[:,1]]

                d = np.sum((a[:, None, :] - a[None, :, ]) ** 2, axis=2)
                k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
            elif(type==1):#テスト時

                if(jyun==0):
                    a = self.ZN1[self.ZN_nums[:, jyun]]
                    d = np.sum((target[:, None, :] - a[None, :, ]) ** 2, axis=2)
                    k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
                    # print(d.shape)
                elif(jyun==1):
                    # print(self.ZN_nums)
                    # print(self.ZN_nums[:,jyun])
                    # print(self.ZN1)
                    a = self.ZN2[self.ZN_nums[:, jyun]]
                    d = np.sum((target[:, None, :] - a[None, :, ]) ** 2, axis=2)
                    k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
                    # print(d.shape)
            return k


        def karnel_jnp(target1,target2,type=0):
            if(type==0):
                k1= kernel_f(target1,0,0)
                k2=kernel_f(target2,0,1)
            elif(type==1):
                k1= kernel_f(target1,1,0)
                k2=kernel_f(target2,1,1)
            # print(target1.shape)
            # print(k1.shape,k2.shape)
            k=jnp.einsum('ij,ij->ij',k1,k2)
            # k=k1
            Y=jnp.einsum('ij,jk->ik',k,self.X)

            # print(Y.shape,k.shape,k[:,:,None].shape)
            YY=Y/jnp.sum(k,axis=1,keepdims=True)
            # print(YY.shape)
            return YY
        def karnel_jnp1(target1,type=1):
            k1= kernel_f(target1,type,1)
            Y=jnp.einsum('ij,jk->ik',k1,self.X)
            YY=Y/jnp.sum(k1,keepdims=True)
            return YY
        def karnel_jnp2(target1,type=1):
            k1= kernel_f(target1,type,1)
            Y=jnp.einsum('ij,jk->ik',k1,self.X)
            YY=Y/jnp.sum(k1,keepdims=True)
            return YY
        #学習時の損失を求める関数
        # @jit
        def E_jnp(target1,target2,epoch=-1,kotti=-1):
            YY = karnel_jnp(target1,target2)
            # YY1=karnel_jnp(target)
            # YY2 = karnel_jnp2(target2)
            loss = {}

            # if(kotti==1):
            #     loss['loss_mse'] = jnp.sum((self.X - YY1) ** 2, axis=1) / 2 / self.D
            #     zn1 = jnp.sum(self.ramuda * target1 ** 2, axis=1) / self.L
            #     loss['loss_L2'] = zn1   # 50,1
            # elif(kotti==2):
            #     loss['loss_mse'] = jnp.sum((self.X - YY1) ** 2, axis=1) / 2 / self.D
            #     zn2 = jnp.sum(self.ramuda * target1 ** 2, axis=1) / self.L
            #     loss['loss_L2'] = zn2 # 50,1

            zn1=jnp.sum(self.ramuda * target1 ** 2, axis=1) / self.L
            zn2 = jnp.sum(self.ramuda * target2 ** 2, axis=1) / self.L
            # print(zn1.shape,self.X_num,self.L)
            # exit()
            # zn2=jnp.zeros((self.X_num))

            loss['loss_L2'] = jnp.sum(zn1)/self.X1_num+jnp.sum(zn2)/self.X2_num
            loss['loss_mse'] = jnp.sum((self.X - YY) ** 2, axis=1) / 2 / self.D/self.X_num

            loss['loss'] = jnp.sum(loss['loss_mse'] + loss['loss_L2'])

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
            #print(sum(self.ZN1))
            dx2 = jax.grad(E_jnp, argnums=1)(self.ZN1,self.ZN2)

            self.ZN2=self.ZN2-self.eta*(dx2)

            #データの保存
            if(epoch%self.meyasu==0):
                #損失の保存
                E_jnp(self.ZN1,self.ZN2,epoch)
                print(self.history['loss'][epoch])
                # print(self.history['loss_mse'][epoch])
                # print(self.history['loss_L2'][epoch])
                # exit()
                #znを用いて生成したYの保存
                ans=karnel_jnp(self.ZN1,self.ZN2)
                self.history['y_zn'][epoch] =ans
                #zkを用いて生成したYの保存
                zk1,zk2=self.make_zk()
                # print('zk1,zk2')
                # print(zk1.shape,zk2.shape)
                ans2 = karnel_jnp(zk1,zk2,1)

                resolution = ans2.reshape(self.KK, self.KK, 3)
                self.history['y_zk'][epoch]=ans2
                self.history['y_zk_wire'][epoch]=resolution
                #潜在空間ZNの保存

                self.history['zn1'][epoch] = self.ZN1
                self.history['zn2'][epoch] = self.ZN2

        #曲面
        def save_data(name,data,syurui):
            np.save(self.sitaikoto+'/' + str(self.basyo) + '/'+syurui+'/'+name, data)

        data=[self.history[i] for i in self.name]
        for i in range(len(self.name)):
            save_data(self.name[i],data[i],'data')


        zn_min = np.zeros((2, 2))
        zn_max = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                if (i == 0):
                    # print(self.ZN1.shape,j)
                    zn_min[i][j] = np.min(self.ZN1[:, j])
                    zn_max[i][j] = np.max(self.ZN1[:, j])
                elif (i == 1):
                    zn_min[i][j] = np.min(self.ZN2[:, j])
                    zn_max[i][j] = np.max(self.ZN2[:, j])

            zk_x = np.linspace(zn_min[i][0], zn_max[i][0], self.KK)
            zk_y = np.linspace(zn_min[i][1], zn_max[i][1], self.KK)
            m_x, m_y = np.meshgrid(zk_x, zk_y)
            m_x = m_x.reshape(-1)
            m_y = m_y.reshape(-1)
            if (i == 0):
                z1 = np.concatenate((m_x[:, None], m_y[:, None]), axis=1)
                # print(z1.shape)
            elif (i == 1):
                z2 = np.concatenate((m_x[:, None], m_y[:, None]), axis=1)

        plt.scatter(z1[:,0], z1[:,1], c='r')
        plt.show()
        plt.cla()
        plt.scatter(z2[:, 0], z2[:, 1], c='r')
        plt.show()
        print('hai')



