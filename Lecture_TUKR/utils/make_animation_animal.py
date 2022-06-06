from matplotlib import animation
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys

print('jiiioji')

print(sys.path)
print(os.path.abspath(__file__))
print(os.getcwd()[:-5])

moto=os.getcwd()[:-5]+'iki/'
sys.path.append(moto)
print(sys.path)



X1_name=['dove','cock','duck','w_duck','owl','hawk','eagle','crow','fox','dog','wolf','cat','tiger','lion','horse','zebra','cattle']
X2_name=['small','medium','large','nocturnality','two_legs','four_legs','hair','hoof','mane','wing','stripe','hunt','run','fly','swim','domestic','herbivorous','carnivore','canidae','felidae','pet']
X1_num=17
X2_num=21
epochs=500
wariai=11
doko=46
frame_epochs=epochs//2

baisu=10
# k_size=100
# kk_size=int(k_size**0.5)
uk_num=15
vk_num=5

d_size=3
l_size=2

data_D=1

e=np.zeros((epochs))
e_seisoku=np.zeros((epochs))
e_loss=np.zeros((epochs))




data={}
name = ['loss', 'loss_mse', 'loss_L2', 'y_zn', 'y_zk', 'y_zk_wire', 'zn1','zn2', 'realx','realx1','realx2','realx3','zk1','zk2','heatmap','heatmap_place']
e_type=['loss','loss_mse','loss_L2']

data={}
sitaikoto='karnel_wo_tiisakusitai'
sitaikoto='data_range_2'
sitaikoto='sakou_no_ans'
sitaikoto='sigma_large'
sitaikoto='sigma_small'
sitaikoto='kantu'
sitaikoto='kansei'
sitaikoto='animal_kantu'
ukr_type=moto+'tukr_animal'
print('-------------------')
if(os.path.exists(sitaikoto)):
    if (os.path.exists(sitaikoto)+'/'+str(doko)):
        print(1)
print('=================')
def load_data(ukr_type,sitaikoto,doko,name):
    return np.load(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/data/'+name+'.npy')
for i in name:
    data[i]=load_data(ukr_type,sitaikoto,doko,i)
    print(i)
    print(data[i].shape)


f = open(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/settings.txt','r')
print(f)
settings= f.readlines()
print(settings[8])


animal_color=['black','gray','rosybrown','firebrick','red','tomato','saddlebrown','bisque','orange','yellow','greenyellow','forestgreen','lime','aquamarine','darkslategray','dodgerblue','slategrey']#17
print(len(animal_color))
type_color=['black','gray','rosybrown','firebrick','red','tomato','saddlebrown','bisque','orange','yellow','greenyellow','forestgreen','lime','aquamarine','darkslategray','dodgerblue','slategrey','royalblue','blue','indigo','magenta']#21
print(len(type_color))

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
    y_zk_hani=np.zeros((data_D))
    for j in range(data_D):
        re=np.max(data['realx'][:,j])-np.min(data['realx'][:,j])
        reso=np.max(data['y_zn'][i,:,j])-np.min(data['y_zn'][i,:,j])
        y_zk_hani[j]=np.max((re,reso))
        y_zk_hani[j]=y_zk_hani[j]
    hirosa = np.max(data['realx3'][:]) - np.min(data['realx3'][:])
    iro=(data['realx3'][:] - np.min(data['realx3'][:]))/hirosa

    ax.set_box_aspect((y_zk_hani[0],y_zk_hani[1] ,y_zk_hani[2]))
    ax.scatter(data['realx'][:,0],data['realx'][:,1],data['realx'][:,2],c=iro)
    ax.scatter(data['y_zn'][i*baisu,:,0], data['y_zn'][i*baisu,:,1], data['y_zn'][i*baisu,:,2], color='b')
    return fig,

def animate_y_zk(i):
    plt.cla()
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    ax.set_xlabel('y_zk1')
    ax.set_ylabel('y_zk2')
    ax.set_zlabel('y_zk3')
    y_zk_hani=np.zeros((3))
    for j in range(data_D):
        re=np.max(data['realx'][:,j])-np.min(data['realx'][:,j])
        reso=np.max(data['y_zk'][i,:,j])-np.min(data['y_zk'][i,:,j])
        y_zk_hani[j]=np.max((re,reso))
        y_zk_hani[j]=y_zk_hani[j]
    hirosa = np.max(data['realx3'][:]) - np.min(data['realx3'][:])
    iro=(data['realx3'][:] - np.min(data['realx3'][:]))/hirosa

    ax.set_box_aspect((y_zk_hani[0],y_zk_hani[1] ,y_zk_hani[2]))

    ax.scatter(data['realx'][:,0],data['realx'][:,1],data['realx'][:,2],c=iro)
    ax.scatter(data['y_zk'][i*baisu,:,0], data['y_zk'][i*baisu,:,1], data['y_zk'][i*baisu,:,2], color='b')
    return fig,

def animate_wire_zk(i):
    plt.cla()
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    ax.set_xlabel('zk1')
    ax.set_ylabel('zk2')
    ax.set_zlabel('zk3')

    resolution = data['y_zk_wire'][i*baisu]

    y_zk_hani=np.zeros((3))
    for j in range(data_D):
        re=np.max(data['realx'][:,j])-np.min(data['realx'][:,j])
        reso=np.max(resolution[:, :, j].reshape(-1))-np.min(resolution[:, :, j].reshape(-1))
        y_zk_hani[j]=np.max((re,reso))
        y_zk_hani[j]=y_zk_hani[j]
    ax.set_box_aspect((y_zk_hani[0],y_zk_hani[1] ,y_zk_hani[2]))
    hirosa=np.max(data['realx3'][:])-np.min(data['realx3'][:])
    #print()
    iro=(data['realx3'][:] - np.min(data['realx3'][:]))/hirosa
    # iro=np.tile(iro,X1_num)

    ax.plot_wireframe(resolution[:, :, 0], resolution[:, :, 1], resolution[:, :, 2], color='b',
                      linewidth=0.3)
    #print(data['realx'].shape,iro.shape)
    ax.scatter(data['realx'][:,0],data['realx'][:,1],data['realx'][:,2],c=iro)
    return fig,

def animate_zn1(i):
    plt.cla()
    ax.axes.set_aspect('equal')
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    plt.xlim(np.min(data['zn1'][i*baisu,:,0]), np.max(data['zn1'][i*baisu,:,0]))  # x軸の範囲
    plt.ylim(np.min(data['zn1'][i*baisu,:,1]), np.max(data['zn1'][i*baisu,:,1]))  # y軸の範囲
    hirosa=np.max(data['realx1'][:])-np.min(data['realx1'][:])
    #iro=(data['realx1'][:] - np.min(data['realx1'][:]))/hirosa
    plt.scatter(data['zn1'][i*baisu,:,0],data['zn1'][i*baisu,:,1],c=animal_color)
    #print('fff')
    for j in range(X1_num):
        plt.annotate(X1_name[j],(data['zn1'][i*baisu,j,0],data['zn1'][i*baisu,j,1]))
    #plt.text(data['zn1'][i*baisu,:,0],data['zn1'][i*baisu,:,1],'1aaaaaaaaaaaaaaaaaaaa')
    return fig,
def animate_zn2(i):
    plt.cla()
    ax.axes.set_aspect('equal')
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    plt.xlim(np.min(data['zn2'][i*baisu,:,0]), np.max(data['zn2'][i*baisu,:,0]))  # x軸の範囲
    plt.ylim(np.min(data['zn2'][i*baisu,:,1]), np.max(data['zn2'][i*baisu,:,1]))  # y軸の範囲
    hirosa=np.max(data['realx2'][:])-np.min(data['realx2'][:])
    iro=(data['realx2'][:] - np.min(data['realx2'][:]))/hirosa
    plt.scatter(data['zn2'][i*baisu,:,0],data['zn2'][i*baisu,:,1],c=type_color)
    for j in range(X2_num):
        plt.annotate(X2_name[j],(data['zn2'][i*baisu,j,0],data['zn2'][i*baisu,j,1]))
    return fig,
def animate_zn1(i):
    plt.cla()
    ax.axes.set_aspect('equal')
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    plt.xlim(np.min(data['zn1'][i*baisu,:,0]), np.max(data['zn1'][i*baisu,:,0]))  # x軸の範囲
    plt.ylim(np.min(data['zn1'][i*baisu,:,1]), np.max(data['zn1'][i*baisu,:,1]))  # y軸の範囲
    hirosa=np.max(data['realx1'][:])-np.min(data['realx1'][:])
    #iro=(data['realx1'][:] - np.min(data['realx1'][:]))/hirosa
    plt.scatter(data['zn1'][i*baisu,:,0],data['zn1'][i*baisu,:,1],c=animal_color)
    #print('fff')
    for j in range(X1_num):
        plt.annotate(X1_name[j],(data['zn1'][i*baisu,j,0],data['zn1'][i*baisu,j,1]))
    #plt.text(data['zn1'][i*baisu,:,0],data['zn1'][i*baisu,:,1],'1aaaaaaaaaaaaaaaaaaaa')
    return fig,
def animate_zk1(i):
    plt.cla()
    ax.axes.set_aspect('equal')
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    plt.xlim(np.min(data['zk1'][i*baisu,:,:]), np.max(data['zk1'][i*baisu,:,:]))  # x軸の範囲
    plt.ylim(np.min(data['zk1'][i*baisu,:,:]), np.max(data['zk1'][i*baisu,:,:]))  # y軸の範囲
    hirosa=np.max(data['realx1'][:])-np.min(data['realx1'][:])
    iro=(data['realx1'][:] - np.min(data['realx1'][:]))/hirosa
    plt.scatter(data['zk1'][i*baisu,:,0],np.zeros(vk_num),c='r')
    return fig,
def animate_zk2(i):
    plt.cla()
    ax.axes.set_aspect('equal')
    plt.suptitle(settings[8]+settings[9]+settings[10]+settings[11],fontsize='8')  # タイトル
    plt.xlim(np.min(data['zk2'][i*baisu,:,:]), np.max(data['zk2'][i*baisu,:,:]))  # x軸の範囲
    plt.ylim(np.min(data['zk2'][i*baisu,:,:]), np.max(data['zk2'][i*baisu,:,:]))  # y軸の範囲
    hirosa=np.max(data['realx2'][:])-np.min(data['realx2'][:])
    iro=(data['realx2'][:] - np.min(data['realx2'][:]))/hirosa
    plt.scatter(data['zk2'][i*baisu,:,0],np.zeros(uk_num),c='r')
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

def draw_zn1_heatmap(name,i):
    plt.figure()
    # ax.axes.set_aspect('equal')
    plt.suptitle(name+'\n',fontsize='14')  # タイトル
    plt.xlim(np.min(data['zn1'][epochs-1,:,0]), np.max(data['zn1'][epochs-1,:,0]))  # x軸の範囲
    plt.ylim(np.min(data['zn1'][epochs-1,:,1]), np.max(data['zn1'][epochs-1,:,1]))  # y軸の範囲
    hirosa=np.max(data['realx1'][:])-np.min(data['realx1'][:])
    #iro=(data['realx1'][:] - np.min(data['realx1'][:]))/hirosa
    #plt.scatter(data['zn1'][epochs-1,:,0],data['zn1'][epochs-1,:,1],c=animal_color)
    # y = np.arange(100)
    cm = plt.get_cmap("Reds")
    hirosa=np.max(data['heatmap'][i,:])-np.min(data['heatmap'][i,:])
    iro=(data['heatmap'][i,:] - np.min(data['heatmap'][i,:]))/hirosa

    # print(hirosa)
    # print(iro)
    # exit()
    plt.scatter(data['heatmap_place'][ i,:, 0],data['heatmap_place'][i, :, 1], color=cm(iro),s=1000)

    #print('fff')
    for j in range(X1_num):
        plt.annotate(X1_name[j],(data['zn1'][epochs-1,j,0],data['zn1'][epochs-1,j,1]))
    #plt.text(data['zn1'][i*baisu,:,0],data['zn1'][i*baisu,:,1],'1aaaaaaaaaaaaaaaaaaaa')
    plt.savefig(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/'+name+'.png')

print(data['zk1'].shape)

print()

print(222)
# print('start wire_zk')
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ani = animation.FuncAnimation(fig, animate_wire_zk, init_func=init,
#                               frames=epochs//baisu, interval=100, blit=True)
# ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/wire_zk.mp4', writer="ffmpeg")
#
# #
# print('start y_zn')
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ani = animation.FuncAnimation(fig, animate_y_zn, init_func=init,
#                               frames=epochs//baisu, interval=100, blit=True)
#
# ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/y_zn.mp4', writer="ffmpeg")
# print('start y_zk')
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ani = animation.FuncAnimation(fig, animate_y_zk, init_func=init,
#                               frames=epochs//baisu, interval=100, blit=True)
#
# ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/y_zk.mp4', writer="ffmpeg")

# print('start zn1')
# fig = plt.figure()
# ax=fig.add_subplot(1,1,1)
# ani = animation.FuncAnimation(fig, animate_zn1, init_func=init,
#                               frames=epochs//baisu, interval=100, blit=True)
# ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/zn1.mp4', writer="ffmpeg")
#
# print('start zn2')
# fig = plt.figure()
# ax=fig.add_subplot(1,1,1)
# ani = animation.FuncAnimation(fig, animate_zn2, init_func=init,
#                               frames=epochs//baisu, interval=100, blit=True)
# ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/zn2.mp4', writer="ffmpeg")


# print('start zk1')
# fig = plt.figure()
# ax=fig.add_subplot(1,1,1)
# ani = animation.FuncAnimation(fig, animate_zk1, init_func=init,
#                               frames=epochs//baisu, interval=100, blit=True)
# ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/zk1.mp4', writer="ffmpeg")
#
# print('start zk2')
# fig = plt.figure()
# ax=fig.add_subplot(1,1,1)
# ani = animation.FuncAnimation(fig, animate_zk2, init_func=init,
#                               frames=epochs//baisu, interval=100, blit=True)
# ani.save(ukr_type+'/'+sitaikoto+'/'+str(doko)+'/zk2.mp4', writer="ffmpeg")
print('start e')
#
for i in e_type:
    graph(data[i],i,wariai)

for i,target_num in enumerate(X2_name):
    draw_zn1_heatmap(target_num,i)



print('owari')

