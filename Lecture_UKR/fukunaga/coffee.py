import numpy as np
import os


def load_date(retlabel_coffee=False, retlabel_feature=False):
    datestore_name = 'datestore/coffee'
    file_name = 'features.txt'

    directory_path = os.path.join(os.path.dirname(__file__), datestore_name)
    file_path = os.path.join(directory_path, file_name)

    x = np.loadtxt(file_path)

    return_objects = [x]

    if retlabel_coffee:
        label_name = 'labels_coffee.txt'
        label_path =os.path.join(directory_path, label_name)
        label_coffee =np.genfromtxt(label_path, dtype=str)
        return_objects.append(label_coffee)

    if retlabel_feature:
        label_name ='labels_feature.txt'
        label_path =os.path.join(directory_path, label_name)
        label_feature =np.genfromtxt(label_path, dtype=str)
        return_objects.append(label_feature)

    return return_objects