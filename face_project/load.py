import numpy as np
from PIL import Image
# img = np.array(Image.open('/Users/furukawashuushi/Desktop/-5/A_01_-05.jpg'))
# print(img.ndim)
# print(img.shape)
import os

def load_angle_resized_data():
    datastore_name = '../datastore/Angle_resized/'
    dir_list = os.listdir(datastore_name)
    #file_name = '/-5/A_01_-05.jpg'

    # directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    # file_path = os.path.join(directory_path, file_name)

    dir_name = dir_list[0]
    user_name = '/A_01_'
    img = []
    file_list = os.listdir(datastore_name + dir_name)
    # print(dir_name)
    for file_name in dir_list:
        if '-' in file_name:
            if '-5' == file_name:
                image = np.array(Image.open(datastore_name + file_name + user_name + '-05' + '.jpg'))
                img.append(image)
            else:
                image = np.array(Image.open(datastore_name + file_name + user_name + file_name + '.jpg'))
                img.append(image)


        elif '0' == file_name:
            image = np.array(Image.open(datastore_name + file_name + user_name + file_name + '.jpg'))
            img.append(image)

        else:
            if '5' == file_name:
                image = np.array(Image.open(datastore_name + file_name + user_name + '+05' + '.jpg'))
                img.append(image)
            else:
                image = np.array(Image.open(datastore_name + file_name + user_name + '+' + file_name + '.jpg'))
                img.append(image)

    # img = cv2.imread(datastore_name + file_name)
    #
    # print(img)
    # plt.imshow(img)
    # plt.show()
    return np.array(img)
def load_angle_resized_same_angle_data():
    datastore_name = '../datastore/Angle_resized/'

    dir_list = os.listdir(datastore_name)
    # print(os.listdir(path='.'))
    # exit()
    #file_name = '/-5/A_01_-05.jpg'

    # directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    # file_path = os.path.join(directory_path, file_name)

    # dir_name = dir_list[0]
    dir_name = str(0)

    img = []
    file_list = os.listdir(datastore_name + dir_name)
    for user_name in file_list:
        image = np.array(Image.open(datastore_name + dir_name + '/' + user_name))
        img.append(image)
    return np.array(img)
# print(load_angle_resized_same_angle_data())
def load_angle_resized_data_TUKR():
    datastore_name = '../datastore/Angle_resized/'
    dir_list = os.listdir(datastore_name)
    #file_name = '/-5/A_01_-05.jpg'
    user_list = os.listdir(datastore_name+'-5/')
    # directory_path = os.path.join(os.path.dirname(__file__), datastore_name)
    # file_path = os.path.join(directory_path, file_name)

    dir_name = dir_list[0]
    # user_name = user_list[0]
    img = []
    # file_list = os.listdir(datastore_name + dir_name)
    # print(user_list[0][0:5])
    # print(user_list)
    # for i in range(90):
    #     user_name = user_list[i][0:5]
    #     print(user_name)
    # exit()
    for i in range(90):
        name = user_list[i][0:5]
        for file_name in dir_list:
            if '-' in file_name:
                if '-5' == file_name:
                    image = np.array(Image.open(datastore_name + file_name + '/' + name + '-05' + '.jpg'))
                    img.append(image)
                else:
                    image = np.array(Image.open(datastore_name + file_name + '/' +name + file_name + '.jpg'))
                    img.append(image)


            elif '0' == file_name:
                image = np.array(Image.open(datastore_name + file_name +'/' + name + file_name + '.jpg'))
                img.append(image)

            else:
                if '5' == file_name:
                    image = np.array(Image.open(datastore_name + file_name +'/' + name + '+05' + '.jpg'))
                    img.append(image)
                else:
                    image = np.array(Image.open(datastore_name + file_name +'/' + name + '+' + file_name + '.jpg'))
                    img.append(image)

    # img = cv2.imread(datastore_name + file_name)
    #
    # print(img)
    # plt.imshow(img)
    # plt.show()
    IMG = np.array(img)
    IMG = np.array(list(np.array_split(IMG, len(user_list))))
    import matplotlib.pyplot as plt
    # for i in range(90):
    #     for j in range(33):
    #         plt.imshow(IMG[i,j],cmap='gray')
    #         plt.show()
    #     exit()

    return IMG

# print(load_angle_resized_data_TUKR().shape)
