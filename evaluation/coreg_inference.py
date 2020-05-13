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
from bxs633.model.my_data_generator import *
from bxs633.model.metrics import *
from bxs633.evaluation.helper_functions import *

""" Co-reg models' inference and save predicted volumes
   :param num_modalities: number of modalities in co-registered data. 3 or 4 
   :param gt_dir: ground truth data directory path
   :param input_dir: test data directory path
   :param model_file: file path to trained model
   :param output_dir: directory path to save output directory """
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    K.set_image_data_format("channels_first")
    
    num_modalities = int(sys.argv[1])
    gt_dir = sys.argv[2]
    input_dir = sys.argv[3]
    model_file = sys.argv[4]
    output_dir = sys.argv[5]

    if(num_modalities == 3):
        loss_func = dice_coefficient_multilable_loss
        eval_metric = dice_coef_multilabel
    else:
        loss_func = dice_coefficient_balanced_avg_loss
        eval_metric = dice_coef_balanced_avg
    

    test_data = get_data(input_dir)
    test_ids = list(range(0, len(test_data)))
    # Parameters
    params = {'batch_size': 1,'n_classes': 5,'n_channels': 1,'shuffle': False, 'slices':128, 'height':128, 'length':128}
    test_generator = DataGenerator(test_ids, test_data, image_path=input_dir, mask_path=gt_dir,to_fit=False, **params)
    avg_whole, avg_core, avg_enhancing = run_inference(loss_func, eval_metric, model_file,output_dir,test_generator, gt_dir, test_data)
    print('Avg Whole: ' + str(avg_whole))
    print('Avg Core: ' + str(avg_core))
    print('Avg Enhancing: ' + str(avg_enhancing))

if __name__ == '__main__':
    main()