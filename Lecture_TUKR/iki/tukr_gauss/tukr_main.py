from tukr import TUKR
import numpy as np
import os

#変数の設定
X1_num=20
X2_num=10
X3_num=1
Z1_num=X1_num
Z2_num=X2_num

X_num=X1_num*X2_num*X3_num
Z_num=X_num

K_num=100
d_num=3
z_num=1

sigma=np.log(X_num)
sigma=0.1
# sigma=0.5
resolution=0.000001


ramuda=0.005
eta=0.3
nb_epoch=200
meyasu=1
x_range=1
data_type='sin(X1)+sin(X2)'
dotti='random'
dotti='kankei_data'
dotti='kanzen_random'
dotti='tensor_random'
dotti='tensor_random'
sitaikoto='karnel_wo_tiisakusitai'
sitaikoto='sigma_small'
sitaikoto='kantu'
sitaikoto='kansei'
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
elif(dotti=='tensor_random'):
    X1 = np.random.uniform(low=-x_range, high=x_range, size=X1_num)
    X2 = np.random.uniform(low=-x_range, high=x_range, size=X2_num)
    #X3=X1[:,None]**2-X2[None:,]**2

    X1=X1.reshape(-1,1)
    X1=np.tile(X1,X2_num)
    X2=np.tile(X2,(X1_num,1))
    # print(X1.shape,X2.shape)
    # print(X1)
    # print(X2)
    print(X1.shape,X2.shape)
    X=np.concatenate([X1[:,:,None],X2[:,:,None]],axis=2)

    X3=X[:,:,0]**2-X[:,:,1]**2
    print(X.shape,X3.shape)
    # print(X3)
    # print(X3.shape)
    # print(X.shape)
    # print(X[:,:,0])
    # print(X)
    X=np.concatenate([X,X3[:,:,None]],axis=2)
    print(X.shape)
    # exit()
    # print(X.shape)
    # exit()


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
    print('self.K=', K_num, file=f)
    print('self.KK=', int(K_num**0.5), file=f)
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
    print('self.K=', K_num, file=f)
    print('self.KK=', int(K_num**0.5), file=f)
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
a=TUKR(X,K_num,z_num,resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo,sitaikoto)


a.fit()

print('fe')
print('どこはここです')



print(basyo)

