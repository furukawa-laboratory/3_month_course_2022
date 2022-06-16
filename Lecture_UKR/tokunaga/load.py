import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image


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

    #dir_name = dir_list[16]
    dir_name = '0'
    img = []
    img_filename = []
    file_list = os.listdir(datastore_name + dir_name)
    for file_name in file_list:
        image = np.array(Image.open(datastore_name + dir_name + '/' + file_name))
        img.append(image)
        img_filename.append(file_name[2:4])
    # img = cv2.imread(datastore_name + file_name)
    #
    # print(img)
    # plt.imshow(img)
    # plt.show()

    return np.array(img), np.array(img_filename)

def load_one_angle_resized_data():
    datastore_name = 'datastore/Angle_resized/'
    dir_list = os.listdir(datastore_name)
    #file_name = '/-5/A_01_-05.jpg'

    # directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    # file_path = os.path.join(directory_path, file_name)

    user_name = '/A_01_'
    img = []
    img_file = []
    for file_name in dir_list:
        if '-' in file_name:
            if '-5' == file_name:
                image = np.array(Image.open(datastore_name + file_name + user_name + '-05' + '.jpg'))
                img.append(image)
                img_file.append(file_name)
            else:
                image = np.array(Image.open(datastore_name + file_name + user_name + file_name + '.jpg'))
                img.append(image)
                img_file.append(file_name)

        elif '0' == file_name:
            image = np.array(Image.open(datastore_name + file_name + user_name + file_name + '.jpg'))
            img.append(image)
            img_file.append(file_name)
        else:
            if '5' == file_name:
                image = np.array(Image.open(datastore_name + file_name + user_name + '+05' + '.jpg'))
                img.append(image)
                img_file.append(file_name)
            else:
                image = np.array(Image.open(datastore_name + file_name + user_name + '+' + file_name + '.jpg'))
                img.append(image)
                img_file.append(file_name)

    # img = cv2.imread(datastore_name + file_name)
    #
    # print(img)
    # plt.imshow(img)
    # plt.show()
    return np.array(img), np.array(img_file)

if __name__ == '__main__':
    img = load_angle_resized_data()
    # print(type(img))