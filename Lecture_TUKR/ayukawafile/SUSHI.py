import numpy as np
import os


def load_data(retlabel_sushi=True):
    datastore_name = 'datastore/SHSHI'
    # file_name = 'preprocessed_data.txt'
    file_name = 'sushi'

    directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    file_path = os.path.join(directory_path, file_name)

    x = np.loadtxt(file_path)

    return_objects = [x]

    if retlabel_sushi:
        label_name = 'SushiName'
        label_path = os.path.join(directory_path, label_name)
        label_sushi = np.genfromtxt(label_path, dtype=str)
        return_objects.append(label_sushi)

    # if retlabel_feature:
    #     label_name = 'labels_feature.txt'
    #     label_path = os.path.join(directory_path, label_name)
    #     label_feature = np.genfromtxt(label_path, dtype=str)
    #     return_objects.append(label_feature)

    # print(np.array(return_objects).shape)
    return return_objects