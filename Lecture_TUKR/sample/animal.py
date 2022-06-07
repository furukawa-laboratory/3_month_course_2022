import numpy as np
import os
from ccp_viewer import TUKR_ccp_viewer


def load_data(retlabel_animal=True, retlabel_feature=False):
    datastore_name = 'datastore'
    file_name = 'features.txt'

    directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    file_path = os.path.join(directory_path, file_name)

    x = np.loadtxt(file_path)

    return_objects = [x]

    if retlabel_animal:
        label_name = 'labels_animal.txt'
        label_path = os.path.join(directory_path, label_name)
        label_animal = np.genfromtxt(label_path, dtype=str)
        return_objects.append(label_animal)

    if retlabel_feature:
        label_name = 'labels_feature.txt'
        label_path = os.path.join(directory_path, label_name)
        label_feature = np.genfromtxt(label_path, dtype=str)
        return_objects.append(label_feature)

    return return_objects

if __name__ == "__main__":  # このファイルを実行した時のみ
    data = load_data(retlabel_animal=True, retlabel_feature=True)
    X = data[0]
    animal_label = data[1]
    feature_label = data[2]

    u = np.load('learning result/u_history.npy')
    v = np.load('learning result/v_history.npy')
    y = np.load('learning result/Y_history.npy')
    zeta_u = np.load('learning result/zetau_history.npy')
    zeta_v = np.load('learning result/zetav_history.npy')
    viewer = TUKR_ccp_viewer(X, y, u, v, fig_size=None, label1=animal_label, label2=feature_label, button_label=None,
                 title_text_1="animal map", title_text_2="feature map", zeta_1=zeta_u, zeta_2=zeta_v)
    viewer.draw_map()
