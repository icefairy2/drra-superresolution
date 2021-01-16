import os
import sys

import cv2
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import add, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.optimizer_v2.adam import Adam

NR_IMAGES_PER_BATCH = 1
DEFAULT_IMAGE_SIZE = 256

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
input_img = Input(shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))

l2 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)

l3 = MaxPooling2D(padding='same')(l2)
l3 = Dropout(0.3)(l3)
l5 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l3)

l6 = MaxPooling2D(padding='same')(l5)
l7 = Conv2D(256, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l6)

l8 = UpSampling2D()(l7)

l10 = Conv2D(128, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l8)

l11 = add([l5, l10])
l12 = UpSampling2D()(l11)
l14 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l12)

l15 = add([l14, l2])

# chan = 3, for RGB
decoded = Conv2D(3, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l15)

# Create our network
super_resolution_model = Model(input_img, decoded)

super_resolution_model.summary()
# install https://www2.graphviz.org/Packages/stable/windows/10/cmake/Release/x64/graphviz-install-2.44.1-win64.exe
# I also had to run the command 'dot -c' with administrator privileges (Windows)
plot_model(super_resolution_model, to_file='model.png')

adam = Adam(lr=0.0003)
super_resolution_model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])


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


def read_image_data(path, shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)):
    images = []
    for filename in os.listdir(path):
        image = cv2.imread(os.path.join(path, filename))
        if image is not None:
            image_resized = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
            images.append(image_resized)
    return images


def train_model(high_res_array_train, low_res_array_train, high_res_array_test, low_res_array_test):
    assert len(high_res_array_train) == len(low_res_array_train), "Train data value and label length not equal!"
    assert len(high_res_array_test) == len(low_res_array_test), "Test data value and label length not equal!"

    super_resolution_model.fit(low_res_array_train, high_res_array_train,
                               epochs=NR_EPOCHS,
                               batch_size=NR_IMAGES_PER_BATCH,
                               shuffle=True,
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

    # Pick a random image to visualize
    image_index = np.random.randint(0, len(low_res_train_data) - 1)

    print("Predicting random image {0}...".format(image_index))
    example_image_lr = np.reshape(low_res_array_train[image_index], [1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3])
    predicted_image = super_resolution_model.predict(example_image_lr)

    # Visualize example image
    cv2.imshow('Low resolution image', low_res_array_train[image_index])
    cv2.imshow('High resolution image', high_res_array_train[image_index])
    cv2.imshow('Predicted image', predicted_image[0].astype('uint8'))
    cv2.waitKey(0)
