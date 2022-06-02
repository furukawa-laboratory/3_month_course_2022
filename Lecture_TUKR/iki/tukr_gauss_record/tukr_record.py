import numpy as np
import matplotlib.pyplot as plt
import jax,jaxlib
import jax.numpy as jnp

import os
class TUKR:
    def __init__(self, X, K_num,latent_dim, resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo,sitaikoto,X1_num,X2_num,X1,X2,ccp_num): # 引数はヒントになるかも！
        #引数を用いたselfの宣言
        self.X = X

        self.X_num, self.D = X.shape
        self.nn=int(self.X_num**0.5)
        self.L = latent_dim
        self.K=K_num
        self.KK=int(self.K**0.5)
        # self.KK=int(self.K**0.5)
        self.resolution = resolution
        self.X1_num=X1_num
        self.X2_num=X2_num
        self.ccp_num=ccp_num

        self.ZN1=np.random.normal(0,0.5,(self.X_num,self.L))
        self.ZN1= np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X_num,self.L))
        self.ZN2 = np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X_num, self.L))
        #record用
        self.ZN1=np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X1_num,self.L))
        self.ZN2 = np.random.uniform(low=-self.resolution, high=self.resolution, size=(self.X2_num, self.L))
        print(type(self.ZN1),self.ZN1.shape)

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
        self.name = ['loss', 'loss_mse', 'loss_L2', 'y_zn', 'y_zk', 'y_zk_wire', 'zn1','zn2', 'realx','realx1','realx2','zk1','zk2','y_Z']
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

        self.history['zk1']=np.zeros((nb_epoch,self.K,self.L))
        self.history['zk2'] = np.zeros((nb_epoch, self.K,self.L))


        self.history['y_Z1']=np.zeros((nb_epoch,ccp_num,ccp_num))#ccp用のZ1空間に表示するためのy
        self.history['y_Z2'] = np.zeros((nb_epoch, ccp_num,ccp_num))#ccp用のZ2空間に表示するためのy
        self.history['y_Z'] = np.zeros((nb_epoch, ccp_num*ccp_num, ccp_num*ccp_num))  # ccp用のZ2空間に表示するためのy

        #self.history['zn'][0]=self.ZN
        self.history['realx']=self.X
        self.history['realx1'] = X1
        self.history['realx2']=X2



    def make_zk(self,epoch):
        zn_min=np.zeros((2,self.L))
        zn_max=np.zeros((2,self.L))
        for i in range(2):
            for j in range(self.L):
                if(i==0):
                    #print(self.ZN1.shape,j)
                    zn_min[i][j]= np.min(self.ZN1[:,j])
                    zn_max[i][j]= np.max(self.ZN1[:,j])
                elif(i==1):
                    zn_min[i][j]= np.min(self.ZN2[:,j])
                    zn_max[i][j]= np.max(self.ZN2[:,j])

        z1 = np.linspace(zn_min[0][0], zn_max[0][0], int(self.K**0.5))
        z1=np.tile(z1,int(self.K**0.5))
        z1=z1[:,None]
        z2 = np.linspace(zn_min[1][0], zn_max[1][0], int(self.K**0.5))
        z2 = np.tile(z2, int(self.K ** 0.5))
        z2=np.sort(z2)
        z2 = z2[:, None]
        # print(self.ZN1.shape,self.ZN2.shape)
        # exit()
        self.history['zk1'][epoch]=z1
        self.history['zk2'][epoch] = z2
        return z1,z2

    def fit(self,):
        #テスト時に入力がznのときのyを求める関数

        def kernel_f(target,type=0,jyun=-1,):
            if(type==0):#学習時
                a=target[self.ZN_nums[:,jyun]]
                d = np.sum((a[:, None, :] - a[None, :, ]) ** 2, axis=2)
                k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
            elif(type==1):#テスト時

                if(jyun==0):
                    a = self.ZN1[self.ZN_nums[:,jyun]]
                    self.bb=target
                    d = np.sum((target[:, None, :] - a[None, :, ]) ** 2, axis=2)
                    k = np.exp(-1 / (2 * self.sigma ** 2) * d)

                elif(jyun==1):
                    a = self.ZN2[self.ZN_nums[:, jyun]]
                    self.bbb=target
                    d = np.sum((target[:, None, :] - a[None, :, ]) ** 2, axis=2)
                    k = np.exp(-1 / (2 * self.sigma ** 2) * d)

            return k


        def karnel_jnp(target1,target2,type=0):

            if(type==0):
                k1= kernel_f(target1,0,0)
                k2=kernel_f(target2,0,1)
            elif(type==1):
                k1= kernel_f(target1,1,0)
                k2=kernel_f(target2,1,1)
            elif(type==2):
                k1= kernel_f(target1,1,0)
                k2=kernel_f(target2,1,1)

                k = jnp.einsum('ij,kj->ikj', k1, k2)
                # k=k1
                Y = jnp.einsum('ikj,jd->ikd', k, self.X)
                # print(k1.shape,k2.shape,k.shape)
                # print(Y.shape)
                # print('h')
                # exit()
                YY = Y / jnp.sum(k, axis=2, keepdims=True)

                return YY

            k=jnp.einsum('ij,ij->ij',k1,k2)
            # k=k1
            Y=jnp.einsum('ij,jk->ik',k,self.X)

            YY=Y/jnp.sum(k,axis=1,keepdims=True)

            return YY

        #学習時の損失を求める関数

        def E_jnp(target1,target2,epoch=-1,kotti=-1):
            YY = karnel_jnp(target1,target2)
            loss = {}
            zn1=jnp.sum(self.ramuda * target1 ** 2, axis=1) / self.L
            zn2 = jnp.sum(self.ramuda * target2 ** 2, axis=1) / self.L

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

                #znを用いて生成したYの保存
                ans=karnel_jnp(self.ZN1,self.ZN2)
                self.history['y_zn'][epoch] =ans
                #zkを用いて生成したYの保存
                zk1,zk2=self.make_zk(epoch)
                # zk1=self.ZN1_calc
                # zk2=self.ZN2_calc

                #ans2 = karnel_jnp(1,1,1)
                ans2 = karnel_jnp(zk1, zk2, 1)
                print(ans2.shape)
                resolution = ans2.reshape(self.KK, self.KK, 3)
                self.history['y_zk'][epoch]=ans2
                self.history['y_zk_wire'][epoch]=resolution
                #潜在空間ZNの保存

                self.history['zn1'][epoch] = self.ZN1
                self.history['zn2'][epoch] = self.ZN2

        def make_z():
            zn_min = np.zeros((2, self.L))
            zn_max = np.zeros((2, self.L))
            for i in range(2):
                for j in range(self.L):
                    if (i == 0):
                        # print(self.ZN1.shape,j)
                        zn_min[i][j] = np.min(self.ZN1[:, j])
                        zn_max[i][j] = np.max(self.ZN1[:, j])
                    elif (i == 1):
                        zn_min[i][j] = np.min(self.ZN2[:, j])
                        zn_max[i][j] = np.max(self.ZN2[:, j])

            z1 = np.linspace(zn_min[0][0], zn_max[0][0], self.ccp_num)
            z1 = np.tile(z1, self.ccp_num)
            z1 = z1[:, None]
            z2 = np.linspace(zn_min[1][0], zn_max[1][0], self.ccp_num)
            z2 = np.tile(z2, self.ccp_num)
            z2 = np.sort(z2)
            z2 = z2[:, None]

            # self.history['z1'][epoch] = z1
            # self.history['z2'][epoch] = z2
            return z1, z2

        def test(self,):
            z1,z2=make_z()
            print(z1.shape,z2.shape)
            ans = karnel_jnp(z1, z2, 2)
            ans=np.sum(ans,axis=2)
            self.history['y_Z']=ans


        test(self,)

        #曲面
        def save_data(name,data,syurui):
            np.save(self.sitaikoto+'/' + str(self.basyo) + '/'+syurui+'/'+name, data)

        data=[self.history[i] for i in self.name]
        for i in range(len(self.name)):
            save_data(self.name[i],data[i],'data')


        zn_min=np.zeros((2,self.L))
        zn_max=np.zeros((2,self.L))
        for i in range(2):
            for j in range(self.L):
                if(i==0):
                    #print(self.ZN1.shape,j)
                    zn_min[i][j]= np.min(self.ZN1[:,j])
                    zn_max[i][j]= np.max(self.ZN1[:,j])
                elif(i==1):
                    zn_min[i][j]= np.min(self.ZN2[:,j])
                    zn_max[i][j]= np.max(self.ZN2[:,j])

        z1 = np.linspace(zn_min[0][0], zn_max[0][0], self.K)
        z1=z1[:,None]
        z2 = np.linspace(zn_min[1][0], zn_max[1][0], self.K)
        z2 = z2[:, None]
        print(zn_min)
        print(zn_max)
        # print('hai')
        print(self.bb)
        print(self.bbb)
        # print(self.b_ue)
        # print(self.b_sita)
        # exit()


