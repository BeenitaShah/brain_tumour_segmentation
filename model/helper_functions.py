import os
import numpy as np
from glob import glob
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from my_data_generator import *
from keras.backend.tensorflow_backend import set_session
import warnings
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.callbacks.tensorboard_v1 import TensorBoard

""" Create training, validation and test data loaders 
    :param input_dir: input data directory path
    :param gt_dir: ground truth directory path 
    :return: training, validation and test generators """
def load_data(input_dir, gt_dir):
    #Split data into training,val and test sets
    all_file_paths = list(glob(input_dir + '*'))
    file_names = [(i.split('/'))[-1] for i in all_file_paths]
    print(len(file_names))
    print(file_names)

    training,test = file_names[:round(len(file_names)*0.8)] ,file_names[round(len(file_names)*0.8):]
    training,val = training[:round(len(training)*0.8)] ,training[round(len(training)*0.8):]

    training_ids = list(range(0, len(training)))
    print(len(training_ids))
    # Parameters
    params = {'batch_size': 1,'n_classes': 5,'n_channels': 1,'shuffle': True, 'slices':128, 'height':128, 'length':128}
    training_generator = DataGenerator(training_ids, training, image_path=input_dir, mask_path=gt_dir,**params)

    val_ids = list(range(0, len(val)))
    print(len(val_ids))
    # Parameters
    params = {'batch_size': 1,'n_classes': 5,'n_channels': 1,'shuffle': True, 'slices':128, 'height':128, 'length':128}
    val_generator = DataGenerator(val_ids, val, image_path=input_dir, mask_path=gt_dir,**params)

    test_ids = list(range(0, len(test)))
    print(len(test_ids))
    # Parameters
    params = {'batch_size': 1,'n_classes': 5,'n_channels': 1,'shuffle': True, 'slices':128, 'height':128, 'length':128}
    test_generator = DataGenerator(test_ids, test, image_path=input_dir, mask_path=gt_dir,**params)

    return training_generator, val_generator, test_generator