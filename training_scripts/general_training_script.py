import os
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam, Adadelta, SGD, Nadam
from keras.layers.merge import concatenate
from keras.backend.tensorflow_backend import set_session
import warnings
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.callbacks.tensorboard_v1 import TensorBoard
import sys
from bxs633.training_scripts.helper_functions import *
from bxs633.model.helper_functions import *
from bxs633.model.unet import *
from bxs633.model.my_data_generator import *


""" Train UNet under optimal hyper-parameters using soft Dice Loss and general Dice metric
   :param input_dir: data directory path
   :param gt_dir: ground truth directory path
   :param file_name: file_name prefix for all files of this run
   :param output_dir: parent output directory path where model, csv and logging will be saved """
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    K.set_image_data_format("channels_first")

    input_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    file_name = sys.argv[3]
    output_dir = sys.argv[4]

    if not os.path.exists(output_dir + 'checkpoints/'):
        os.makedirs(output_dir + 'checkpoints/')
    if not os.path.exists(output_dir + 'tensorflow_dir/'):
        os.makedirs(output_dir + 'tensorflow_dir/')
    if not os.path.exists(output_dir + 'csv_logs/'):
        os.makedirs(output_dir + 'csv_logs/')

    training_generator, val_generator, test_generator = load_data(input_dir, gt_dir)

    optim = Adam(learning_rate=0.00001)
    training_func(optim,200,output_dir,file_name,training_generator,val_generator)


if __name__ == '__main__':
    main()