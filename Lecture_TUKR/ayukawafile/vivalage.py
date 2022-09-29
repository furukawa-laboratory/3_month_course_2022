import numpy as np
import os


def load_data(retlabel_drink=True, retlabel_joukyou=True):
    datastore_name = 'datastore/raw_data'
    file_name = '001'

    directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    file_path = os.path.join(directory_path, file_name)

    x = np.loadtxt(file_path)

    return_objects = [x]

    # print(return_objects)
    #
    # f = return_objects
    # print("what up")
    # print(x[0][ :, :, None])
    #
    # print("what if")

    if retlabel_drink:
        label_name = 'drink_label'
        label_path = os.path.join(directory_path, label_name)
        label_drink = np.genfromtxt(label_path, dtype=str)
        return_objects.append(label_drink)

    if retlabel_joukyou:
        label_name = 'joukyou_label'
        label_path = os.path.join(directory_path, label_name)
        label_joukyou = np.genfromtxt(label_path, dtype=str)
        return_objects.append(label_joukyou)

    print(np.array(return_objects[0]).shape)
    # print(return_objects)
    return return_objects

load_data()