import os
import sys

import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.vis_utils import plot_model

from adamlrm import AdamLRM

# Constants for training hyperparameters

DEFAULT_IMAGE_SIZE = 512

NR_EPOCHS = 100
NR_IMAGES_PER_BATCH = 10
VALIDATION_SPLIT = 0.2

# Constants for command line arguments
ARG_X8 = "x8"
ARG_MILD = "mild"
ARG_WILD = "wild"
ARG_DIFFICULT = "difficult"

# Constants for data paths
HIGH_RES_DATA_PATH_TRAIN = "data/DIV2K_train_HR"
HIGH_RES_DATA_PATH_TEST = "data/DIV2K_valid_HR"

LOW_RES_DATA_PATH_TRAIN_X8 = "data/DIV2K_train_LR_x8"
LOW_RES_DATA_PATH_TEST_X8 = "data/DIV2K_valid_LR_x8"

LOW_RES_DATA_PATH_TRAIN_MILD = "data/DIV2K_train_LR_mild"
LOW_RES_DATA_PATH_TEST_MILD = "data/DIV2K_valid_LR_mild"

LOW_RES_DATA_PATH_TRAIN_WILD = "data/DIV2K_train_LR_wild"
LOW_RES_DATA_PATH_TEST_WILD = "data/DIV2K_valid_LR_wild"

LOW_RES_DATA_PATH_TRAIN_DIFFICULT = "data/DIV2K_train_LR_difficult"
LOW_RES_DATA_PATH_TEST_DIFFICULT = "data/DIV2K_valid_LR_difficult"

# The model which we based on SRCNN but modified in the best way we could
# http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
SRCNN_modified = Sequential()

SRCNN_modified.add(Conv2D(filters=128, kernel_size=(9, 9), activation='relu', padding="same", use_bias=True,
                          input_shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), name="layer1"))
SRCNN_modified.add(
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same", use_bias=True, name="layer2"))
SRCNN_modified.add(
    Conv2D(filters=3, kernel_size=(5, 5), activation='relu', padding="same", use_bias=True, name="layer3"))

# We implemented variable learning rates per layer in order to try to achieve good performance
lr_multiplier = {
    'layer1': 0.001,
    'layer2': 0.0001,
    'layer3': 0.00001
}

# In order to achieve variable learning rates, we based ourselves on the implementation of akinoux
# for a modified Adam optimizer: https://github.com/akionux/AdamLRM
opt = AdamLRM(lr=0.01, lr_multiplier=lr_multiplier)

# We use MSE as a loss function like the original SRCNN
SRCNN_modified.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
SRCNN_modified.summary()

# This is a method to automatically plot the model architecture
# For Windows:
# - you need to install Graphviz from
#   https://www2.graphviz.org/Packages/stable/windows/10/cmake/Release/x64/graphviz-install-2.44.1-win64.exe
# - install pydot via: pip install pydot
# - run a cmd as administrator and run the command 'dot -c'
plot_model(SRCNN_modified, to_file='model.png')


def define_low_res_paths(type):
    """
    This method parses the command line argument and chooses the path to the corresponding low resolution image category
    :param type: the low resolution image category, default ARG_X8='x8'
    :return: the path to the low resolution images
    """
    if type is None or type == ARG_X8:
        print("Using bicubic x8 downscaling...")
        return LOW_RES_DATA_PATH_TRAIN_X8, LOW_RES_DATA_PATH_TEST_X8
    if type == ARG_MILD:
        print("Using realistic mild x4 downscaling...")
        return LOW_RES_DATA_PATH_TRAIN_MILD, LOW_RES_DATA_PATH_TEST_MILD
    if type == ARG_WILD:
        print("Using realistic wild x4 downscaling...")
        return LOW_RES_DATA_PATH_TRAIN_WILD, LOW_RES_DATA_PATH_TEST_WILD
    if type == ARG_DIFFICULT:
        print("Using realistic difficult x4 downscaling...")
        return LOW_RES_DATA_PATH_TRAIN_DIFFICULT, LOW_RES_DATA_PATH_TEST_DIFFICULT


def read_image_data(path, shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)):
    """
    This method reads in all the images from a given path and returns then in an array; resizing at a common
    shape is also done here at reading
    :param path: the folder path to the images
    :param shape: (width, height) the desired size for the image
    :return: list of images with equal sizes
    """
    images = []
    for filename in os.listdir(path):
        image = cv2.imread(os.path.join(path, filename))
        if image is not None:
            # We added the interpolation parameter because the default compression left artifacts in the
            # resized high resolution images
            image_resized = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
            images.append(image_resized)
    return images


def train_model(high_res_array_train, low_res_array_train):
    """
    This method trains our model
    :param high_res_array_train: the label data, the high resolution image feature array
    :param low_res_array_train: the input data, the low resolution image feature array
    """
    assert len(high_res_array_train) == len(low_res_array_train), "Train data value and label length not equal!"
    assert len(high_res_array_test) == len(low_res_array_test), "Test data value and label length not equal!"

    SRCNN_modified.fit(low_res_array_train, high_res_array_train,
                       epochs=NR_EPOCHS,
                       batch_size=NR_IMAGES_PER_BATCH,
                       shuffle=True,
                       validation_split=VALIDATION_SPLIT)


if __name__ == "__main__":
    low_res_type = ARG_X8

    # Read command line arguments
    if len(sys.argv) > 0:
        low_res_type = sys.argv[1]

    low_res_train_path, low_res_test_path = define_low_res_paths(low_res_type)

    print("Reading low resolution images...")
    low_res_train_data = read_image_data(low_res_train_path)
    low_res_test_data = read_image_data(low_res_test_path)

    print("Reading high resolution images...")
    high_res_train_data = read_image_data(HIGH_RES_DATA_PATH_TRAIN)
    high_res_test_data = read_image_data(HIGH_RES_DATA_PATH_TEST)

    print("Converting to numpy arrays...")
    low_res_array_train = np.divide(np.array(low_res_train_data), 255, dtype='float32')
    high_res_array_train = np.divide(np.array(high_res_train_data), 255, dtype='float32')

    low_res_array_test = np.divide(np.array(low_res_test_data), 255, dtype='float32')
    high_res_array_test = np.divide(np.array(high_res_test_data), 255, dtype='float32')

    print("Training model...")
    train_model(high_res_array_train, low_res_array_train)

    # Pick a random image to visualize
    image_index = np.random.randint(0, len(low_res_test_data) - 1)

    print("Predicting random image {0}...".format(image_index))
    example_image_lr = np.array([low_res_array_test[image_index]])
    predicted_image = SRCNN_modified.predict(example_image_lr)
    predicted_image *= 255

    # Visualize example image
    cv2.imshow('Low resolution image', low_res_test_data[image_index])
    cv2.imshow('High resolution image', high_res_test_data[image_index])
    cv2.imshow('Predicted image', predicted_image[0].astype('uint8'))
    cv2.waitKey(0)
