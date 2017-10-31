import numpy as np
import pandas as pd
import bson
import seaborn as sns
import matplotlib.pyplot as plt
import os, io
import threading

import vgg16
from bsoniterator import BSONIterator

from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder
from keras import optimizers
from keras.preprocessing.image import Iterator, ImageDataGenerator, load_img
from keras import backend as K
from keras.models import load_model

def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

data_dir = "./data/"

# load files
train_bson_path = os.path.join(data_dir, "train.bson")
num_train_products = 7069896
test_bson_path = os.path.join(data_dir, "test.bson")
num_test_products = 1768182

categories_df = pd.read_csv("categories.csv", index_col=0)
cat2idx, idx2cat = make_category_tables()

train_offsets_df = pd.read_csv("train_offsets.csv", index_col=0)
train_images_df = pd.read_csv("train_images.csv", index_col=0)
val_images_df = pd.read_csv("val_images.csv", index_col=0)

train_bson_file = open(train_bson_path,'rb')

num_classes = 5270
num_train_images = len(train_images_df)
num_val_images = len(val_images_df)
batch_size = 128

# Tip: use ImageDataGenerator for data augmentation and preprocessing.
lock = threading.Lock()
train_datagen = ImageDataGenerator()
train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                         num_classes, train_datagen, lock,
                         batch_size=batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                       num_classes, val_datagen, lock,
                       batch_size=batch_size, shuffle=True)

#TRAINING----------------------------------------------------------
#model = vgg16.create_model(class_num=num_classes, input_shape=(180, 180, 3))
model = load_model('./log/2017_1030_0208/1000epochs_final_save')
opt = optimizers.SGD(lr=0.0001, momentum=0.9)
vgg16.train(model, train_gen, val_gen, optimizer=opt, epochs=4000)
