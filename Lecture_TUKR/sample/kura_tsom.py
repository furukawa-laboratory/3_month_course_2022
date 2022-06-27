import numpy as np



def load_kura_tsom(xsamples,ysamples,ret_truez=False,probabilistic=False):
    z1=np.linspace(-1,1,xsamples)
    z2=np.linspace(-1,1,ysamples)
    # z1 = 2.0 * np.random.rand(xsamples) - 1.0
    # z2 = 2.0 * np.random.rand(ysamples) - 1.0

    zz1,zz2=np.meshgrid(z2,z1)
    z=zz1**2-zz2**2

    true_z = (z1, z2)

    #確率分布として扱うかどうか
    if probabilistic is True:
        X=np.exp(z)/np.sum(np.exp(z))

    else:
        X = np.concatenate((zz1[:, :, np.newaxis], zz2[:, :, np.newaxis], z[:, :, np.newaxis]), axis=2)

    if ret_truez is True:
        return (X,true_z)
    else:
        return X



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xsamples = 20
    ysamples = 25

    x, true_z = load_kura_tsom(xsamples, ysamples, ret_truez=True)



    fig = plt.figure()
    ax=Axes3D(fig)
    ax.scatter(x[:,:,0],x[:,:,1],x[:,:,2])
    plt.show()