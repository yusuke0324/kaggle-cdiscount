import numpy as np
import os

from datetime import datetime as dt
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger
from keras.models import Model
from keras.models import load_model

from keras.callbacks import LambdaCallback

def create_model(class_num=3, freeze_layer_num=15, input_shape=(224, 224, 3)):
    '''
    return VGG16 with imagenet weights. Their first 15 (dafault) layers are freezed for finetuning.

    ARG:
        class_num: number of classes you need to classify
        freeze_layer_num: number of first layers to be freezed for finetuning
        input_shape: input image shape (height, weight, channel_num)
    '''
    input_tensor = Input(dtype='float32', shape=input_shape)
    # load weighted VGG16
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=input_shape)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(class_num, activation='softmax'))

    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))

    for layer in model.layers[:freeze_layer_num]:
        layer.trainable = False

    return model

def _epochOutput(epoch, logs):

    print("Finished epoch: " + str(epoch))
    print(logs)

def _get_date_str():
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y_%m%d_%H%M')
    return tstr

def _make_dir(dir_name):
    if not os.path.exists('dir_name'):
        os.makedirs(dir_name)

def train(model, train_gen, val_gen, optimizer='adam', log_dir='./log', batch_size=15, epochs=10, steps_per_epoch=10, validation_steps=10):
    sub_dir = _get_date_str()
    log_dir = log_dir + '/' + sub_dir
    # make log dir
    _make_dir(log_dir)
    # saved model path
    fpath = log_dir + '/weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'

    # callback
    tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    batchLogCallback = LambdaCallback(on_epoch_end=_epochOutput)
    csv_logger = CSVLogger(log_dir + '/vgg_training.log')
    callbacks = [batchLogCallback, csv_logger]

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit_generator(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        validation_data=val_gen,
        workers=8,
        validation_steps=validation_steps,
        callbacks=callbacks,
        )
    
    model.save(log_dir + '/' + str(epochs) + 'epochs_final_save')

