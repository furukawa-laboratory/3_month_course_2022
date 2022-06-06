import numpy as np
import matplotlib.pyplot as plt
import jax,jaxlib
import jax.numpy as jnp

import os
class TUKR:
    def __init__(self, X, uk_num,vk_num,latent_dim, K_num,resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo,sitaikoto,X1_num,X2_num,ZN_nums): # 引数はヒントになるかも！
        #引数を用いたselfの宣言
        self.X = X

        self.X_num, self.D = X.shape
        self.nn=int(self.X_num**0.5)
        self.L = latent_dim
        # self.K=K_num
        self.uk_num=uk_num
        self.vk_num = vk_num
        self.K_num=K_num

        # self.KK=int(self.K**0.5)
        self.resolution = resolution
        self.X1_num=X1_num
        self.X2_num=X2_num

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

        self.ZN_nums= ZN_nums




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
        self.name = ['loss', 'loss_mse', 'loss_L2', 'y_zn', 'y_zk', 'y_zk_wire', 'zn1','zn2', 'realx','realx1','realx2','realx3','zk1','zk2','heatmap','heatmap_place']
        # 学習過程記録用 historyの初期化
        self.history = {}
        self.history['zn1'] = np.zeros((nb_epoch, self.X1_num, self.L))
        self.history['zn2'] = np.zeros((nb_epoch, self.X2_num, self.L))
        self.history['y_zn'] = np.zeros((nb_epoch, self.X_num, self.D))
        self.history['y_zk'] = np.zeros((nb_epoch, self.uk_num*self.vk_num, self.D))
        self.history['y_zk_wire']=np.zeros((nb_epoch,self.uk_num,self.vk_num,self.D))
        self.history['loss']=np.zeros((self.nb_epoch))
        self.history['loss_mse']=np.zeros((self.nb_epoch))
        self.history['loss_L2'] = np.zeros((self.nb_epoch))
        self.history['realx']=np.zeros((self.X_num,self.D))
        self.history['realx1'] = np.zeros((self.X1_num, self.D))
        self.history['realx2'] = np.zeros((self.X2_num, self.D))
        self.history['realx3']=np.zeros((self.X_num,self.D))

        self.history['zk1']=np.zeros((nb_epoch,self.uk_num,self.L))
        self.history['zk2'] = np.zeros((nb_epoch, self.vk_num,self.L))
        self.history['heatmap']=np.zeros((self.X2_num,self.K_num**2,self.D))
        self.history['heatmap_place'] = np.zeros((self.X2_num,self.K_num ** 2, self.L))

        #self.history['zn'][0]=self.ZN
        self.history['realx']=self.X




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

        z1 = np.linspace(zn_min[0][0], zn_max[0][0], self.uk_num)
        self.history['zk1'][epoch]=z1[:,None]
        z1=np.tile(z1,self.vk_num)
        z1=np.sort(z1)
        z1=z1[:,None]
        z2 = np.linspace(zn_min[1][0], zn_max[1][0], self.vk_num)
        self.history['zk2'][epoch] = z2[:,None]
        z2 = np.tile(z2, self.uk_num)
        # z2=np.sort(z2)
        z2 = z2[:, None]

        return z1,z2

    def fit(self,):
        #テスト時に入力がznのときのyを求める関数

        def kernel_f(target,type=0,jyun=-1,):
            if(type==0):#学習時
                a=target[self.ZN_nums[:,jyun]]
                # print(a.shape)
                # print(a.shape)
                d = np.sum((a[:, None, :] - a[None, :, ]) ** 2, axis=2)
                # print(d.shape)
                k = jnp.exp(-1 / (2 * self.sigma ** 2) * d)
                # print(k.shape)
                # exit()
            elif(type==1):#テスト時

                if(jyun==0):

                    a = self.ZN1[self.ZN_nums[:,jyun]]
                    # self.b_ue=np.sort(a,axis=0)
                    self.bb=target
                    # b=target[::-1]
                    # b=a
                    #b=a[::-1]
                    d = np.sum((target[:, None, :] - a[None, :, ]) ** 2, axis=2)
                    k = np.exp(-1 / (2 * self.sigma ** 2) * d)

                elif(jyun==1):
                    a = self.ZN2[self.ZN_nums[:, jyun]]
                    # self.b_sita = np.sort(a,axis=0)[::self.X2_num]
                    # self.b_sita=np.tile(self.b_sita,(self.X1_num,1))
                    self.bbb=target
                    # b=target[::-1]
                    # b=a
                    # b=a[::-1]
                    d = np.sum((target[:, None, :] - a[None, :, ]) ** 2, axis=2)
                    k = np.exp(-1 / (2 * self.sigma ** 2) * d)
            # print(k.shape)
            return k


        def karnel_jnp(target1,target2,type=0):

            if(type==0):
                k1= kernel_f(target1,0,0)
                k2=kernel_f(target2,0,1)
            elif(type==1):
                k1= kernel_f(target1,1,0)
                k2=kernel_f(target2,1,1)

            k=jnp.einsum('ij,ij->ij',k1,k2)
            # print(k.shape)
            # k=k1
            Y=jnp.einsum('ij,jk->ik',k,self.X)
            # print(Y.shape)

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
            # print(np.sum(self.ZN1))
            # print(np.sum(self.ZN2))
            #データの保存
            if(epoch%self.meyasu==0):
                #損失の保存
                E_jnp(self.ZN1,self.ZN2,epoch)
                print(self.history['loss'][epoch])

                # #znを用いて生成したYの保存
                # ans=karnel_jnp(self.ZN1,self.ZN2)
                # self.history['y_zn'][epoch] =ans
                # #zkを用いて生成したYの保存
                # zk1,zk2=self.make_zk(epoch)
                # # zk1=self.ZN1_calc
                # # zk2=self.ZN2_calc
                #
                # #ans2 = karnel_jnp(1,1,1)
                # ans2 = karnel_jnp(zk1, zk2, 1)
                # print(ans2.shape)
                # resolution = ans2.reshape(self.uk_num, self.vk_num, 3)
                # self.history['y_zk'][epoch]=ans2
                # self.history['y_zk_wire'][epoch]=resolution
                #潜在空間ZNの保存

                self.history['zn1'][epoch] = self.ZN1
                self.history['zn2'][epoch] = self.ZN2

        #曲面
        def feature_test(target_num):
            # k_num = 10
            zk1_min=np.zeros(2)
            zk1_max = np.zeros(2)
            zk1=np.zeros((2,self.K_num))

            for i in range(self.L):
                zk1_min[i]=np.min(self.ZN1[:,i])
                zk1_max[i] = np.max(self.ZN1[:,i])
                # zk1_min[i] = -5
                # zk1_max[i] = 5
                # print(zk1_min[i],zk1_max[i])
                # print(type(zk1_min[i]))
                # print(type(zk1_max[i]))
                # zk1[i] = np.linspace(-10,10, k_num)
                zk1[i] = np.linspace(zk1_min[i],zk1_max[i], self.K_num)
            zk1_mx,zk1_my=np.meshgrid(zk1[0],zk1[0])
            zk1_mx=zk1_mx.reshape((-1,1))
            zk1_my = zk1_my.reshape((-1,1))
            real_zk1=np.concatenate((zk1_mx,zk1_my),axis=1)

            target_zk2=self.ZN2[target_num]
            target_zk2=np.tile(target_zk2,(self.K_num**2,1))
            ans2 = karnel_jnp(real_zk1,target_zk2 ,1)
            self.history['heatmap'][target_num]=ans2
            self.history['heatmap_place'][target_num]=real_zk1


        for i in range(self.X2_num):
            feature_test(i)


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

        z1 = np.linspace(zn_min[0][0], zn_max[0][0], self.uk_num)
        z1=z1[:,None]
        z2 = np.linspace(zn_min[1][0], zn_max[1][0], self.vk_num)
        z2 = z2[:, None]
        print(zn_min)
        print(zn_max)
        print(self.ZN_nums[0:24])
        # print(np.sum(self.history['ZN1']))
        # print('hai')

        # print(self.b_ue)
        # print(self.b_sita)
        # exit()


