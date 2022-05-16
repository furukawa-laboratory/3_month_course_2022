import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import jax,jaxlib
import jax.numpy as jnp
import numpy as np

epochs=200-1
wariai=10
nani='ukr_gauss'
doko=1
frame_epochs=epochs//2
baisu=10
k_size=10000
kk_size=int(k_size**0.5)
n_size=100
d_size=3
l_size=2

y_zk=np.zeros((epochs,k_size,d_size))
y_zn=np.zeros((epochs,n_size,d_size))
y_zk_wire=np.zeros((epochs,kk_size,kk_size,d_size))
zn=np.zeros((epochs,n_size,l_size))

realx=np.zeros((epochs,n_size,d_size))
e=np.zeros((epochs))
e_seisoku=np.zeros((epochs))
e_loss=np.zeros((epochs))




data={}
name = ['loss', 'loss_mse', 'loss_L2', 'y_zn', 'y_zk', 'y_zk_wire', 'zn', 'realx']
e_type=['loss','loss_mse','loss_L2']

data={}
sitaikoto='karnel_wo_tiisakusitai'
sitaikoto='data_range_2'
ukr_type='ukr_gauss'
########################
def load_data(ukr_type,sitaikoto,doko,name):
    return np.load(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/data/'+name+'.npy')
for i in name:
    data[i]=load_data(ukr_type,sitaikoto,doko,i)

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
    y_zk_hani=np.zeros((3))
    for j in range(3):
        re=np.max(realx[:,j])-np.min(realx[:,j])
        reso=np.max(resolution[:, :, j].reshape(-1))-np.min(resolution[:, :, j].reshape(-1))
        y_zk_hani[j]=np.max((re,reso))
        y_zk_hani[j]=int(y_zk_hani[j])
    ax.set_box_aspect((y_zk_hani[0],y_zk_hani[1] ,y_zk_hani[2]))
    hirosa=np.max(realx[:,0])-np.min(realx[:,0])
    iro=(realx[:,0] - np.min(realx[:, 0]))/hirosa
    ax.plot_wireframe(resolution[:,:, 0], resolution[:,:, 1], resolution[:,:, 2], color='b',
                      linewidth=0.3)
    print('reso')
    ax.scatter(realx[:,0],realx[:,1],realx[:,2],c=iro)

def images_y_zn(i,y_zn,realx,):
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('zn1')
    ax.set_ylabel('zn2')
    ax.set_zlabel('zn3')
    resolution = y_zn[i]
    print(resolution.shape)
    hirosa=np.max(realx[:,0])-np.min(realx[:,0])
    iro=(realx[:,0] - np.min(realx[:, 0]))/hirosa
    ax.scatter(y_zn[i:,:,0], y_zn[i:,:, 1],y_zn[i:,:, 2], color='b',
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
print(data['y_zk_wire'].shape)
images_wire_zk(epochs,data['y_zk_wire'],data['realx'])
images_y_zn(epochs,data['y_zn'],data['realx'])
plt.show()
#start e
# wariai=3
# graph(e,'e',wariai)
# graph(e_seisoku,'e_seisoku',wariai)
# graph(e_loss,'e_loss',wariai)
#
# print('owari')
