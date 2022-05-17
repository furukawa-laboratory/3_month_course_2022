import numpy as np
import matplotlib.pyplot as plt

def load_kura_tsom(xsamples, ysamples, missing_rate=None,retz=False):
    z1 = 
    z2 = 

    z1_repeated, z2_repeated = 
    x1 = 
    x2 = 
    x3 = 
    #ノイズを加えたい時はここをいじる,locがガウス分布の平均、scaleが分散,size何個ノイズを作るか
    #このノイズを加えることによって三次元空間のデータ点は上下に動く

    x = 
    truez = 

    if missing_rate == 0 or missing_rate == None:
        if retz:
            return x, truez
        else:
            return x

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xsamples = 
    ysamples = 

    x, truez = load_kura_tsom() 
    
    fig = plt.figure(figsize=[5, 5])
    ax_x = fig.add_subplot(projection='3d')
    ax_x.scatter()
    ax_x.set_title('Generated three-dimensional data')
    plt.show()