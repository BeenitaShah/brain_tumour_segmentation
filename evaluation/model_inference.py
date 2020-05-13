import numpy as np
import os
import glob
from glob import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.backend.tensorflow_backend import set_session
import warnings
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from helper_functions import *
import sys  
from bxs633.evaluation.helper_functions import *


""" Model inference 
   :param input_dir: test data directory path
   :param gt_dir: ground truth directory path
   :param model_weight_file: file path to trained model (.hd5f file)
   :param output_dir: prediction output directory path """
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    K.set_image_data_format("channels_first")

    input_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    model_weight_file = sys.argv[3]
    output_dir = sys.argv[4]

    if not os.path.exists(output_dir + 'checkpoints/'):
        os.makedirs(output_dir + 'checkpoints/')
    if not os.path.exists(output_dir + 'tensorflow_dir/'):
        os.makedirs(output_dir + 'tensorflow_dir/')
    if not os.path.exists(output_dir + 'csv_logs/'):
        os.makedirs(output_dir + 'csv_logs/')

    loss_func = dice_coefficient_loss_gen
    eval_metric = dice_coefficient

    test_data = get_data(input_dir)
    test_ids = list(range(0, len(test_data)))
    print(len(test_ids))
    # Parameters
    params = {'batch_size': 1,'n_classes': 5,'n_channels': 1,'shuffle': False, 'slices':128, 'height':128, 'length':128}
    test_generator = DataGenerator(test_ids, test_data, image_path=input_dir, mask_path=gt_dir,to_fit=False, **params)

    avg_whole, avg_core, avg_enhancing = run_inference(loss_func, eval_metric, model_weight_file, output_dir,test_generator, gt_dir, test_data)
    print('Avg Whole: ' + str(avg_whole))
    print('Avg Core: ' + str(avg_core))
    print('Avg Enhancing: ' + str(avg_enhancing))
    
if __name__ == '__main__':
    main()

