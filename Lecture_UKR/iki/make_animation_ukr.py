from matplotlib import animation
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

epochs=200
wariai=10
nani='ukr_gauss'
doko=2
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
sitaikoto='sakou_no_ans'
sitaikoto='sigma_large'
sitaikoto='sigma_small'
ukr_type='ukr_gauss'
print('-------------------')
if(os.path.exists(sitaikoto)):
    if (os.path.exists(sitaikoto)+'/'+str(doko)):
        print(1)
print('=================')
def load_data(ukr_type,sitaikoto,doko,name):
    return np.load(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/data/'+name+'.npy')
for i in name:
    data[i]=load_data(ukr_type,sitaikoto,doko,i)


f = open(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/settings.txt','r')
print(f)
settings= f.readlines()
print(settings[8])

# ##################################################################################
# # 描画ようのメソッド


def init():
    return fig,

def animate_y_zn(i):
    plt.cla()
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    ax.set_xlabel('y_zn1')
    ax.set_ylabel('y_zn2')
    ax.set_zlabel('y_zn3')
    y_zk_hani=np.zeros((3))
    for j in range(3):
        re=np.max(data['realx'][:,j])-np.min(data['realx'][:,j])
        reso=np.max(data['y_zn'][i,:,j])-np.min(data['y_zn'][i,:,j])
        y_zk_hani[j]=np.max((re,reso))
        y_zk_hani[j]=int(y_zk_hani[j])

    ax.set_box_aspect((y_zk_hani[0],y_zk_hani[1] ,y_zk_hani[2]))

    ax.scatter(data['realx'][:,0],data['realx'][:,1],data['realx'][:,2],color='r')
    ax.scatter(data['y_zn'][i*baisu,:,0], data['y_zn'][i*baisu,:,1], data['y_zn'][i*baisu,:,2], color='b')
    return fig,

def animate_wire_zk(i):
    plt.cla()
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    ax.set_xlabel('zk1')
    ax.set_ylabel('zk2')
    ax.set_zlabel('zk3')

    resolution = data['y_zk_wire'][i*baisu]

    y_zk_hani=np.zeros((3))
    for j in range(3):
        re=np.max(data['realx'][:,j])-np.min(data['realx'][:,j])
        reso=np.max(resolution[:, :, j].reshape(-1))-np.min(resolution[:, :, j].reshape(-1))
        y_zk_hani[j]=np.max((re,reso))
        y_zk_hani[j]=int(y_zk_hani[j])
    ax.set_box_aspect((y_zk_hani[0],y_zk_hani[1] ,y_zk_hani[2]))
    hirosa=np.max(data['realx'][:,0])-np.min(data['realx'][:,0])
    iro=(data['realx'][:,0] - np.min(data['realx'][:, 0]))/hirosa
    ax.plot_wireframe(resolution[:, :, 0], resolution[:, :, 1], resolution[:, :, 2], color='b',
                      linewidth=0.3)
    ax.scatter(data['realx'][:,0],data['realx'][:,1],data['realx'][:,2],c=iro)
    return fig,

def animate_zn(i):
    plt.cla()
    ax.axes.set_aspect('equal')
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    plt.xlim(np.min(data['zn'][i*baisu,:,:]), np.max(data['zn'][i*baisu,:,:]))  # x軸の範囲
    plt.ylim(np.min(data['zn'][i*baisu,:,:]), np.max(data['zn'][i*baisu,:,:]))  # y軸の範囲
    hirosa=np.max(data['realx'][:,0])-np.min(data['realx'][:,0])
    iro=(data['realx'][:,0] - np.min(data['realx'][:, 0]))/hirosa
    plt.scatter(data['zn'][i*baisu,:,0],data['zn'][i*baisu,:,1],c=iro)
    return fig,

def graph(y,name,wariai):
    plt.figure()
    # plt.title(settings[8],settings[9],settings[10],settings[11])  # タイトル
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    epoch=list(y.shape)[0]
    start=epoch//wariai
    x=np.arange(epoch)
    plt.plot(x[start:],y[start:])
    plt.savefig(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/'+name+'.png')






print(222)
print('start wire_zk')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ani = animation.FuncAnimation(fig, animate_wire_zk, init_func=init,
                              frames=epochs//baisu, interval=100, blit=True)
ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/wire_zk.mp4', writer="ffmpeg")


print('start y_zn')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ani = animation.FuncAnimation(fig, animate_y_zn, init_func=init,
                              frames=epochs//baisu, interval=100, blit=True)

ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/y_zn.mp4', writer="ffmpeg")

print('start zn')
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
ani = animation.FuncAnimation(fig, animate_zn, init_func=init,
                              frames=epochs//baisu, interval=100, blit=True)
ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/zn.mp4', writer="ffmpeg")
print('start e')
#
for i in e_type:
    graph(data[i],i,wariai)



print('owari')
