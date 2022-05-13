from ukr_gauss import UKR
import numpy as np
import os

#変数の設定
X_num=100
K_num=100
d_num=3
z_num=2

sigma=np.log(X_num)
sigma=0.5
resolution=0.2
ramuda=0.005
eta=15
nb_epoch=200
meyasu=1
x_range=1

dotti='random'

if(dotti=='random'):
    X1=np.random.uniform(low=-x_range, high=x_range, size=X_num)
    X2=np.random.uniform(low=-x_range, high=x_range,size= X_num)
elif(dotti=='init'):
    X1=np.linspace(-x_range,x_range,X_num) +10#Z1
    X2=np.linspace(-x_range,x_range,X_num)
else:
    exit()
X3=X1**2-X2**2

X=np.concatenate([X1[:,None],X2[:,None]],axis=1)

X=np.concatenate([X,X3[:,None]],axis=1)
#保存用ディレクトリの作成
def make_dict(dict_name,nb_epoch):

    if(os.path.exists(dict_name)):
        #Dataのエポックフォルダがある時
        if (os.path.exists(dict_name+'/' + str(nb_epoch))):
            con = 0
            while (1):
                if (os.path.exists(dict_name+'/' + str(nb_epoch) + '/' + str(con))):
                    con = con + 1
                else:
                    os.mkdir(dict_name+'/' + str(nb_epoch) + '/' + str(con))
                    break
    #Dataのエポックフォルダがない時
        else:
            con = 0
            os.mkdir(dict_name+'/' + str(nb_epoch))
            os.mkdir(dict_name+'/' + str(nb_epoch) + '/' + str(0))
    else:
        os.mkdir(dict_name)
        con = 0
        os.mkdir(dict_name+'/' + str(nb_epoch))
        os.mkdir(dict_name+'/' + str(nb_epoch) + '/' + str(0))

    return con
basyo=make_dict('data',nb_epoch)
make_dict('pic',nb_epoch)
with open('data/' + str(nb_epoch) + '/' + str(basyo) + '/settings.txt', 'w') as f:
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

a=UKR(X,K_num,z_num,resolution,ramuda,eta,sigma,nb_epoch,meyasu,x_range,basyo)


a.fit()

print('fe')
print('どこはここです')
print('henkou_test')
print(basyo)
