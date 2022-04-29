import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import jax,jaxlib
import jax.numpy as jnp
import numpy as np
epochs=200
baisu=1
k_size=10000
kk_size=int(k_size**0.5)
n_size=100
d_size=3
l_size=2
epoch=199
y_zk=np.zeros((epochs,k_size,d_size))
y_zn=np.zeros((epochs,n_size,d_size))
y_zk_wire=np.zeros((epochs,kk_size,kk_size,d_size))
zn=np.zeros((epochs,n_size,l_size))
realx=np.zeros((n_size,d_size))
e=np.zeros((epochs))
e_seisoku=np.zeros((epochs))
e_loss=np.zeros((epochs))

nani='ukr_itiyou'
doko=4
import os
def make_dict(syurui,dict_name,nb_epoch,doko):

    if(os.path.exists(syurui+'/'+dict_name)):
        #Dataのエポックフォルダがある時
        if (os.path.exists(syurui+'/'+dict_name+'/' + str(nb_epoch))):
            if (os.path.exists(syurui+'/'+dict_name+'/' + str(nb_epoch) + '/' + str(doko))):
                pass
            else:
                os.mkdir(syurui+'/'+dict_name+'/' + str(nb_epoch) + '/' + str(doko))

    #Dataのエポックフォルダがない時
        else:
            os.mkdir(syurui+'/'+dict_name+'/' + str(nb_epoch))
            os.mkdir(syurui+'/'+dict_name+'/' + str(nb_epoch) + '/' + str(0))
    else:
        os.mkdir(syurui+'/'+dict_name)
        os.mkdir(syurui+'/'+dict_name+'/' + str(nb_epoch))
        os.mkdir(syurui+'/'+dict_name+'/' + str(nb_epoch) + '/' + str(0))

make_dict(nani,'pic',epochs,doko)



y_zk=np.load(nani+'/data/'+str(epochs*baisu)+'/'+str(doko)+'/y_zk.npy')
y_zn=np.load(nani+'/data/'+str(epochs*baisu)+'/'+str(doko)+'/y_zn.npy')
zn=np.load(nani+'/data/'+str(epochs*baisu)+'/'+str(doko)+'/zn.npy')
realx=np.load(nani+'/data/'+str(epochs*baisu)+'/'+str(doko)+'/realx_0.npy')
y_zk_wire=np.load(nani+'/data/'+str(epochs*baisu)+'/'+str(doko)+'/y_zk_wire.npy')
e=np.load(nani+'/data/'+str(epochs*baisu)+'/'+str(doko)+'/e.npy')
e_seisoku=np.load(nani+'/data/'+str(epochs*baisu)+'/'+str(doko)+'/e_seisoku.npy')
e_loss=np.load(nani+'/data/'+str(epochs*baisu)+'/'+str(doko)+'/e_loss.npy')
history ={} # 値保存用変数

history['y_zn1'], history['y_zn2'], history['y_zn3'] = np.zeros((epochs,n_size)), np.zeros((epochs,n_size)), np.zeros((epochs,n_size))

history['y_zn1']= y_zn[:,:,0]
history['y_zn2']= y_zn[:,:,1]
history['y_zn3']= y_zn[:,:,2]
history['y_zk1'], history['y_zk2'], history['y_zk3'] = np.zeros((epochs,k_size)), np.zeros((epochs,k_size)), np.zeros((epochs,k_size))


history['y_zk1']= y_zk[:,0]
history['y_zk2'] = y_zk[:,1]
history['y_zk3']= y_zk[:,2]

history['zn1'], history['zn2']= np.zeros((epochs,n_size)), np.zeros((epochs,n_size))

history['zn1']= zn[:,:,0]
history['zn2']= zn[:,:,1]


# ##################################################################################
# # 描画ようのメソッド
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

# fig2=plt.figure()
# ax = fig2.add_subplot(1, 1, 1, projection='3d')
import matplotlib.cm as cm
import matplotlib.pylab as pylab
import matplotlib.cm as cm



def init():
    return fig,

def animate_y_zn(i):
    plt.cla()
    ax.set_xlabel('y_zn1')
    ax.set_ylabel('y_zn2')
    ax.set_zlabel('y_zn3')
    ax.scatter(realx[:,0],realx[:,1],realx[:,2],color='r')
    ax.scatter(history['y_zn1'][i], history['y_zn2'][i], history['y_zn3'][i], color='b')
    return fig,

def animate_wire_zk(i):
    plt.cla()
    ax.set_xlabel('zk1')
    ax.set_ylabel('zk2')
    ax.set_zlabel('zk3')
    resolution = y_zk_wire[i]
    hirosa=np.max(realx[:,0])-np.min(realx[:,0])
    iro=(realx[:,0] - np.min(realx[:, 0]))/hirosa
    ax.plot_wireframe(resolution[:, :, 0], resolution[:, :, 1], resolution[:, :, 2], color='b',
                      linewidth=0.3)
    ax.scatter(realx[:,0],realx[:,1],realx[:,2],c=iro)
    return fig,

def animate_zn(i):
    plt.cla()
    cm = plt.get_cmap("Reds")
    hirosa=np.max(realx[:,0])-np.min(realx[:,0])
    iro=(realx[:,0] - np.min(realx[:, 0]))/hirosa
    plt.scatter(history['zn1'][i],history['zn2'][i],c=iro)
    #plt.scatter(history['zn1'][i],history['zn2'][i],color=realx[:,0])
    return fig1,


def images(i,x,history,name):
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    print(str(1))
    a=name+str(1)
    b = name + str(2)
    c = name + str(3)

    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_zlabel(c)
    ax.scatter(x[:,0],x[:,1],x[:,2],color='r')
    ax.scatter(history[a][i], history[b][i], history[c][i], color='b')
    plt.show()
    plt.savefig(nani+'/mp4/'+str(epochs*baisu)+'/'+str(doko)+'/'+name+'.png')

def images_wire_zk(i,y_zk_wire,realx,):
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('zk1')
    ax.set_ylabel('zk2')
    ax.set_zlabel('zk3')
    resolution = y_zk_wire[i]
    print(resolution.shape)
    hirosa=np.max(realx[:,0])-np.min(realx[:,0])
    iro=(realx[:,0] - np.min(realx[:, 0]))/hirosa
    ax.plot_wireframe(resolution[:,:, 0], resolution[:,:, 1], resolution[:,:, 2], color='b',
                      linewidth=0.3)
    print('reso')
    ax.scatter(realx[:,0],realx[:,1],realx[:,2],c=iro)

def graph(y,name,wariai):
    plt.figure()
    epoch=list(y.shape)[0]
    start=epoch//wariai
    x=np.arange(epoch)
    plt.plot(x[start:],y[start:])
    plt.savefig(nani+'/mp4/'+str(epochs*baisu)+'/'+str(doko)+'/'+name+'.png')


# images(epoch,realx,history,'y_zn')
images_wire_zk(epoch,y_zk_wire,realx)

plt.show()
#start e
# wariai=3
# graph(e,'e',wariai)
# graph(e_seisoku,'e_seisoku',wariai)
# graph(e_loss,'e_loss',wariai)
#
# print('owari')
