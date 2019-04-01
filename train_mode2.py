import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from face_dataset import load_dataset, resize_image, IMAGE_SIZE

def select_data(images, labels, img_rows, img_cols,img_channels):



    train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                              random_state=random.randint(0, 100))

    _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                      random_state=random.randint(0, 100))

    print (K.image_dim_ordering())

    train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
    valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
    image_shap = (img_rows, img_cols, img_channels)
    print(train_images.shape[0], 'train samples')
    print(valid_images.shape[0], 'valid samples')
    print(test_images.shape[0], 'test samples')
    train_images = train_images.astype('float32')
    valid_images = valid_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255.0
    valid_images /= 255.0
    test_images /= 255.0
    train_labels = np_utils.to_categorical(train_labels, 2)
    valid_labels = np_utils.to_categorical(valid_labels, 2)
    test_labels = np_utils.to_categorical(test_labels, 2)

    return train_images, train_labels, valid_images, valid_labels




def model(img_rows, img_cols,img_channels,train_images, train_labels, valid_images, valid_labels):
    image_shap = (img_rows, img_cols, img_channels)
    model = Sequential()
    model.add(Convolution2D(
        nb_filter=32,
        nb_row=3,
        nb_col=3,
        border_mode='same',
        input_shape=image_shap
    ))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=32,
        nb_row=3,
        nb_col=3,
        border_mode='same'
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same'
    ))
    model.add(Dropout(0.25))
    model.add(Convolution2D(
        nb_filter=64,
        nb_row=3,
        nb_col=3,
        border_mode='same',
        input_shape=image_shap
    ))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        nb_filter=64,
        nb_row=3,
        nb_col=3,
        border_mode='same',
        input_shape=image_shap
    ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same'
    ))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()

    batch_size = 20
    nb_epoch = 100




    sgd = SGD(lr=0.01, decay=1e-6,
              momentum=0.9, nesterov=True)  
    model.compile(loss='categorical_crossentropy',
                       optimizer=sgd,
                       metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)


    datagen.fit(train_images)

    model.fit_generator(datagen.flow(train_images, train_labels,
                                     batch_size=batch_size),
                        samples_per_epoch=train_images.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(valid_images, valid_labels))




img_rows = 64
img_cols = 64
img_channels = 3


images, labels = load_dataset()
train_images, train_labels, valid_images, valid_labels = select_data(images, labels, img_rows, img_cols,img_channels)
model(img_rows, img_cols,img_channels,train_images, train_labels, valid_images, valid_labels)
