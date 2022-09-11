from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras import backend as K
import keras
import numpy as np

def load_data(d_name):

    if d_name == "cifar10": 
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

    elif d_name == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        num_classes = 100
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

    elif d_name == "mnist":
        img_rows, img_cols = 28, 28

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    elif d_name == "fashion_mnist":
        print("data set MNIST-Fashion")
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        num_classes = 10

        if K.image_data_format() == "channels_first":
            x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
            x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))
        else:
            x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
            x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

    # simple preprocessing
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    return (x_train, y_train), (x_test, y_test)

def get_num_classes(d_name):
    if d_name == "cifar10":
        return 10 
    elif d_name == "cifar100":
        return 100
    elif d_name == "mnist":
        return 10
    elif d_name == "fashion_mnist":
        return 10