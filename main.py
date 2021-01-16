import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import add, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam

from adamlrm import AdamLRM

NR_IMAGES_PER_BATCH = 10
DEFAULT_IMAGE_SIZE = (128, 128)

NR_EPOCHS = 4

ARG_X8 = "x8"
ARG_MILD = "mild"
ARG_WILD = "wild"
ARG_DIFFICULT = "difficult"

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


####################### TODO IMPLEMENT MODEL #############################

"""**ENCODER**"""
input_img = Input(shape=(128, 128, 3))

l1 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)
l2 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)
l3 = MaxPooling2D(padding='same')(l2)
l3 = Dropout(0.3)(l3)
l4 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l3)
l5 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)
l6 = MaxPooling2D(padding='same')(l5)
l7 = Conv2D(32, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l6)
encoder = Model(input_img, l7)

encoder.summary()
encoder.compile(optimizer="Adam", loss="mse")

"""**DECODER**"""
l1 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)
l2 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)
l3 = MaxPooling2D(padding='same')(l2)
l3 = Dropout(0.3)(l3)
l4 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l3)
l5 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)
l6 = MaxPooling2D(padding='same')(l5)
l7 = Conv2D(32, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l6)

l8 = UpSampling2D()(l7)
l9 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l8)
l10 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l9)

l11 = add([l5, l10])
l12 = UpSampling2D()(l11)
l13 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l12)
l14 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l13)
l15 = add([l14, l2])
decoded = Conv2D(3, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l15)

autoencoder = Model(input_img, decoded)
autoencoder_hfenn = Model(input_img, decoded)

# autoencoder.summary()
autoencoder.compile(optimizer="Adam", loss="mse")

""" ANOTHER MODEL from https://github.com/xoraus/Super-Resolution-CNN-for-Image-Restoration """
# define model type
SRCNN = Sequential()

# add model layers
SRCNN.add(
    Conv2D(filters=128, kernel_size=(9, 9), kernel_initializer='glorot_uniform', activation='relu', padding="same",
           use_bias=True, input_shape=(128, 128, 3), name="layer1"))
SRCNN.add(
    Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='glorot_uniform', activation='relu', padding="same",
           use_bias=True, name="layer2"))
SRCNN.add(
    Conv2D(filters=1, kernel_size=(5, 5), kernel_initializer='glorot_uniform', activation='linear', padding="same",
           use_bias=True, name="layer3"))

lr_multiplier = {
    'layer1': 0.0001,  # optimize 'var1*' with a smaller learning rate
    'layer2': 0.0001,
    'layer3': 0.00001 # optimize 'var2*' with a larger learning rate
}

opt = AdamLRM(lr=0.001, lr_multiplier=lr_multiplier)

# compile model
SRCNN.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
SRCNN.summary()


###################################################################


def define_low_res_paths(type=ARG_X8):
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

# TODO: implement the correct conversion to Y

def convert_to_Y(image):
    # convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # create image slice and normalize
    y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255

    return y


def convert_to_RGB(images, temp=None, output=None):
    # copy Y channel back to image and convert to BGR    
    for image in range(len(images)):
        temp[:, :, 0] = image[0, :, :, 0]
        output[image] = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

    return output


def read_image_data(path, shape=(128, 128)):
    images = []
    for filename in os.listdir(path):
        image = cv2.imread(os.path.join(path, filename))
        if image is not None:
            image_resized = cv2.resize(image, shape)
            # In case we want to use the Y channel only
            # image_converted = convert_to_Y(image_resized)
            images.append(image_resized)
    return images


def train_model(high_res_array_train, low_res_array_train, high_res_array_test, low_res_array_test):
    assert len(high_res_array_train) == len(low_res_array_train), "Train data value and label length not equal!"
    assert len(high_res_array_test) == len(low_res_array_test), "Test data value and label length not equal!"

    # autoencoder.fit(low_res_array_train, high_res_array_train,
    #                 epochs=NR_EPOCHS,
    #                 batch_size=NR_IMAGES_PER_BATCH,
    #                 shuffle=True,
    #                 validation_data=(low_res_array_test, high_res_array_test))

    SRCNN.fit(low_res_array_train, high_res_array_train,
              epochs=NR_EPOCHS,
              batch_size=NR_IMAGES_PER_BATCH,
              validation_data=(low_res_array_test, high_res_array_test))


if __name__ == "__main__":
    low_res_type = sys.argv[1]

    low_res_train_path, low_res_test_path = define_low_res_paths(low_res_type)

    print("Reading low resolution images...")
    low_res_train_data = read_image_data(low_res_train_path)
    low_res_test_data = read_image_data(low_res_test_path)

    print("Reading high resolution images...")
    high_res_train_data = read_image_data(HIGH_RES_DATA_PATH_TRAIN)
    high_res_test_data = read_image_data(HIGH_RES_DATA_PATH_TEST)

    print("Converting to numpy arrays...")
    high_res_array_train = np.array(high_res_train_data)
    low_res_array_train = np.array(low_res_train_data)

    high_res_array_test = np.array(high_res_test_data)
    low_res_array_test = np.array(low_res_test_data)

    # with open('data/hr_train.pkl', 'wb') as f:
    #     pickle.dump(high_res_array_train, f)
    #
    # with open('data/hr_test.pkl', 'wb') as f:
    #     pickle.dump(high_res_array_test, f)

    # with open('data/hr_train.pkl', 'rb') as f:
    #     high_res_array_train = pickle.load(f)
    #
    # with open('data/hr_test.pkl', 'rb') as f:
    #     high_res_array_test = pickle.load(f)

    print("Training model...")
    train_model(high_res_array_train, low_res_array_train, high_res_array_test, low_res_array_test)

    print("Predicting images...")
    # predicted_images = autoencoder.predict(low_res_array_train)
    predicted_images = SRCNN.predict(low_res_array_train)
    # predicted_images = convert_to_RGB(predicted_images)

    # Pick a random image to visualize
    image_index = np.random.randint(0, 799)

    # Visualize example image
    fig = plt.figure()

    ax = []
    i = 1

    ax.append(fig.add_subplot(3, 3, i))
    ax[-1].set_title("Low resolution")  # set title
    plt.imshow(low_res_array_train[image_index])

    i += 1
    ax.append(fig.add_subplot(3, 3, i))
    ax[-1].set_title("Predicted")  # set title
    plt.imshow(predicted_images[image_index].astype('uint8'))

    i += 1
    ax.append(fig.add_subplot(3, 3, i))
    ax[-1].set_title("High resolution")  # set title
    plt.imshow(high_res_array_train[image_index])

    plt.show()
