import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def load_animal_data(retlabel_animal=False, retlabel_feature=True):
    datastore_name = 'datastore/animal'
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

def load_coffee_data():
    datastore_name = 'datastore/coffee'
    file_name = 'features.txt'

    directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    file_path = os.path.join(directory_path, file_name)

    x = np.loadtxt(file_path)

    return_objects = [x]

    label_name = 'labels.txt'
    label_path = os.path.join(directory_path, label_name)
    label = np.genfromtxt(label_path, dtype=str)
    return_objects.append(label)

    return return_objects

def load_angle_resized_data():
    datastore_name = 'datastore/Angle_resized/'
    dir_list = os.listdir(datastore_name)
    #file_name = '/-5/A_01_-05.jpg'

    # directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    # file_path = os.path.join(directory_path, file_name)

    dir_name = dir_list[0]
    img = []
    file_list = os.listdir(datastore_name + dir_name)
    for file_name in file_list:
        image = cv2.imread(datastore_name + dir_name + '/' + file_name)
        img.append(image)

    img_gray = np.sum(np.array(img), axis=3) / 3
    # img = cv2.imread(datastore_name + file_name)
    #
    # print(img)
    # plt.imshow(img)
    # plt.show()

    return img_gray


if __name__ == '__main__':
    img = load_angle_resized_data()
    print()