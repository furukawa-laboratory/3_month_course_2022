from tukr_animal import TUKR
import numpy as np
import os
from animal import load_data
data=load_data()
#変数の設定
print(len(data))

X1_num,X2_num=data[0].shape
X3_num=1
X1_name=data[1]
X2_name=data[2]
print(type(X1_name))
print(X2_name)

print(type(data[0]))
print(type(data[0][0]))
print(X1_num,X2_num,X3_num)
# exit()
Z1_num=X1_num
Z2_num=X2_num

X_num=X1_num*X2_num*X3_num


K_num=10#方片の数
uk_num=15
vk_num=5
d_num=3
z_num=2

sigma=np.log(X_num)
sigma=0.1
# sigma=0.5
resolution=0.000001


ramuda=0.0005
eta=1
nb_epoch=500
meyasu=1
x_range=1
data_type='sin(X1)+sin(X2)'
dotti='random'
dotti='kankei_data'
dotti='kanzen_random'
dotti='record_random'
dotti='animal_data'
sitaikoto='karnel_wo_tiisakusitai'
sitaikoto='sigma_small'
sitaikoto='kantu'
sitaikoto='kansei'
sitaikoto='animal_kantu'
syurui='data'

if(dotti=='random'):
    X1=np.random.uniform(low=-x_range, high=x_range, size=X_num)
    X2=np.random.uniform(low=-x_range, high=x_range,size= X_num)
elif(dotti=='init'):
    X1=np.linspace(-x_range,x_range,X_num) +10#Z1
    X2=np.linspace(-x_range,x_range,X_num)
elif(dotti=='kankei_data'):
    X1=np.linspace(-x_range,x_range,X1_num)
    X2 = np.linspace(-x_range, x_range, X2_num)
    print('fafef')
    print(X1[:,None].shape,X2.shape)
    print(X1[:,None,None].shape)
    print(X1[:,None,None].shape)
    print(X2[None,:,None].shape)
    X3=X1[:,None,None]**2-X2[None,:,None]**2 #10,5
    m_x, m_y = np.meshgrid(X1, X2)
    m_x = m_x.reshape(-1)
    m_y = m_y.reshape(-1)
    X_graph_mode=np.concatenate((m_x[:, None], m_y[:, None]), axis=1)
    print(X_graph_mode.shape,X3.reshape(-1)[:,None].shape)
    X_graph_mode=np.concatenate((X_graph_mode,X3.reshape(-1)[:,None]),axis=1)
    print(X1.shape,X2.shape,X3.shape,X_graph_mode.shape)
elif(dotti=='kanzen_random'):
    X1=np.random.uniform(low=-x_range, high=x_range, size=X_num)
    X2=np.random.uniform(low=-x_range, high=x_range,size= X_num)
    X3=X1**2-X2**2

    X=np.concatenate([X1[:,None],X2[:,None]],axis=1) #
    X=np.concatenate([X,X3[:,None]],axis=1)
elif(dotti=='record_random'):
    X1=np.random.uniform(low=-x_range,high=x_range,size=X1_num)
    X2=np.random.uniform(low=-x_range,high=x_range,size=X2_num)
    # X1=np.linspace(-x_range,x_range,X1_num)
    # X2 = np.linspace(-x_range, x_range, X2_num)

    #X3=X1[:,None]**2-X2[None,:]**2 #10,5
    m_x, m_y = np.meshgrid(X1, X2)
    m_x = m_x.reshape(-1)
    m_y = m_y.reshape(-1)
    X3=m_x**2-m_y**2
    X_sin=np.concatenate((m_x[:, None], m_y[:, None]), axis=1)

    X=np.concatenate([X_sin,X3.reshape(-1)[:,None]],axis=1)

    z1=np.arange(X1_num)
    z2 = np.arange(X2_num)
    m_x, m_y = np.meshgrid(z1, z2)
    m_x = m_x.reshape(-1)
    m_y = m_y.reshape(-1)
    z=np.concatenate((m_x[:, None], m_y[:, None]), axis=1)

    X=X+np.random.uniform(low=-x_range*(0.01),high=x_range*(0.01),size=(X_num,d_num))
    import matplotlib.pyplot as plt
    # fig=plt.figure()
    # ax=fig.add_subplot(1,1,1,projection='3d')
    # ax.scatter(X[:,0], X[:,1], X[:,2], color='b')
    # plt.show()
    # X1=np.tile(X1,X2_num)+np.random.uniform(low=-x_range*(0.001),high=x_range*(0.001),size=X_num)
    # X2 = np.tile(X2, X1_num)+np.random.uniform(low=-x_range*(0.001),high=x_range*(0.001),size=X_num)
    # from sklearn.preprocessing import MinMaxScaler
    # X_std = MinMaxScaler().fit_transform(X_sin)
    # x0 = X_std[:, 0].reshape(-1, 1)
    # x1 = X_std[:, 1].reshape(-1, 1)
    # rgb_array = np.concatenate((np.zeros_like(x0), np.zeros_like(x0), x1), axis=1)
    # rgb_array2 = np.concatenate((x0, np.zeros_like(x0), np.zeros_like(x0)), axis=1)
    # #ax.scatter(Z[:, 0], Z[:, 1], s=1, c=rgb_array)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,2,1,projection='3d')
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.scatter(X[:,0], X[:,1], X[:,2], c=rgb_array)
    # ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=rgb_array2)
    # plt.show()
elif(dotti=='animal_data'):
    X=data[0].reshape(X1_num*X2_num,X3_num)
    print(X[0:10])
    z1=np.arange(X2_num)
    z2=np.arange(X1_num)
    m_x,m_y=np.meshgrid(z1,z2)
    m_x=m_x.reshape(-1)
    m_y = m_y.reshape(-1)
    ZN_nums=np.concatenate((m_x[:,None],m_y[:,None]),axis=1)
    ZN_nums=np.flip(ZN_nums,axis=1)
    print(ZN_nums.shape)
    print(ZN_nums[0:10])


    #print(X_kari[0:10])
    #print(X_kari.shape)



else:
    exit()
# X3=X1**2-X2**2
#X3=np.sin(X1)+np.sin(X2)

# X=np.concatenate([X1[:,None],X2[:,None]],axis=1) #
# X=np.concatenate([X,X3[:,None]],axis=1)



#保存用ディレクトリの作成
def make_dict(dict_name,syurui):
    #sitaikoto dictがあったとき
    if(os.path.exists(dict_name)):
        if(os.path.exists(dict_name+'/overview')):
            pass
        else:
            os.mkdir(dict_name+'/overview')
        #順番がある時
        con=0
        while (1):
            if (os.path.exists(dict_name+'/' + str(con) )):
                con = con + 1
            else:
                os.mkdir(dict_name + '/' + str(con))
                os.mkdir(dict_name + '/' + str(con) + '/' + syurui)
                break
        #Dataのエポックフォルダがない時

    else:
        os.mkdir(dict_name)
        if(os.path.exists(dict_name+'/overview')):
            pass
        else:
            os.mkdir(dict_name+'/overview')
        con = 0
        os.mkdir(dict_name+'/' + str(0))
        os.mkdir(dict_name+'/' + str(0)+'/'+syurui )

    return con


basyo=make_dict(sitaikoto,syurui)

with open(sitaikoto+'/overview''/all_settings.txt', 'a') as f:
    print('以下の設定は@'+str(basyo)+'@===================================================',file=f)
    print('一言コメント',file=f)
    print('テストですよーーーーーーーー',file=f)
    print('self.N=', X_num, file=f)
    print('self.D=', d_num, file=f)
    print('self.nn=', int(X_num**0.5), file=f)
    print('self.uk_num=', uk_num, file=f)
    print('self.vk_num=', vk_num, file=f)
    print('self.sigma=', sigma, file=f)
    print('self.resolution=',resolution, file=f)
    print('ramuda=', ramuda, file=f)
    print('self.eta=',eta, file=f)
    print('self.nb_epoch=', nb_epoch, file=f)
    print('meyasu=', meyasu, file=f)
    print('x_range=', x_range, file=f)
    print('data_type=',data_type,file=f)
with open(sitaikoto+'/'+str(basyo)+'/settings.txt', 'w') as f:
    print('以下の設定は@'+str(basyo)+'@===================================================',file=f)
    print('一言コメント',file=f)
    print('テストですよーーーーーーーー',file=f)
    print('self.N=', X_num, file=f)
    print('self.D=', d_num, file=f)
    print('self.nn=', int(X_num**0.5), file=f)
    print('self.uk_num=', uk_num, file=f)
    print('self.vk_num=', vk_num, file=f)
    print('self.sigma=', sigma, file=f)
    print('self.resolution=',resolution, file=f)
    print('ramuda=', ramuda, file=f)
    print('self.eta=',eta, file=f)
    print('self.nb_epoch=', nb_epoch, file=f)
    print('meyasu=', meyasu, file=f)
    print('x_range=', x_range, file=f)
    print('data_type=',data_type,file=f)
with open(sitaikoto+'/overview''/mokuteki.txt', 'w') as f:
    print('xの範囲を−2から2にしたい',file=f)
a=TUKR(X,uk_num,vk_num,z_num,K_num,resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo,sitaikoto,X1_num,X2_num,ZN_nums)


a.fit()

print('fe')
print('どこはここです')



print(basyo)

