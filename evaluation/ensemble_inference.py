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
import sys  
from bxs633.model.my_data_generator import *
from bxs633.evaluation.helper_functions import *

""" Ensemble models' inference and save predicted volumes
   :param gt_dir: ground truth data directory path
   :param t1_dir: t1 trained model's prediction data directory path
   :param t1c_dir: t1c trained model's prediction data directory path
   :param t2_dir: t2 trained model's prediction data directory path
   :param flair_dir: flair trained model's prediction data directory path
   :param agg_4_out_dir: 4 modality, aggregation ensemble prediction output directory path
   :param agg_3_out_dir: 3 modality, aggregation ensemble prediction output directory path
   :param voting_4_out_dir: 4 modality, voting ensemble prediction output directory path
   :param voting_3_out_dir: 3 modality, voting ensemble prediction output directory path """
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    K.set_image_data_format("channels_first")

    gt_dir = sys.argv[1]
    t1_dir = sys.argv[2]
    t1c_dir = sys.argv[3]
    t2_dir = sys.argv[4]
    flair_dir = sys.argv[5]
    agg_4_out_dir = sys.argv[6]
    agg_3_out_dir = sys.argv[7]
    voting_4_out_dir = sys.argv[8]
    voting_3_out_dir = sys.argv[9]

    #Calculate set of files over all prediction data directories
    t1 = list(glob(t1_dir + "*"))
    t1_files = [i.split('/')[-1] for i in t1]
    print(len(t1_files))

    flair = list(glob(flair_dir + "*"))
    flair_files = [i.split('/')[-1] for i in flair]
    print(len(flair_files))

    t1c = list(glob(t1c_dir + "*"))
    t1c_files = [i.split('/')[-1] for i in t1c]
    print(len(t1c_files))

    t2 = list(glob(t2_dir + "*"))
    t2_files = [i.split('/')[-1] for i in t2]

    file_names = list(set(t1_files) & set(flair_files) & set(t1c_files) & set(t2_files))
    print(len(file_names))

    #Calculate average Dice ensemble scores using 4 modalities and aggregation ensemble mechanism - save final segmentation map
    agg_4_whole,agg_4_core,agg_4_enh = get_agg_4(file_names, t1_dir, t1c_dir, t2_dir, flair_dir, gt_dir, agg_4_out_dir)
    print('---------------------------------------')
    print('Ensemble 4, aggregation results: ')
    print('Whole: ' + str(agg_4_whole))
    print('Core: ' + str(agg_4_core))
    print('Enhancing: ' + str(agg_4_enh))
    print('---------------------------------------')

    #Calculate average Dice ensemble scores using 3 modalities and aggregation ensemble mechanism - save final segmentation map
    agg_3_whole,agg_3_core,agg_3_enh = get_agg_3(file_names,t1c_dir, t2_dir, flair_dir, gt_dir, agg_3_out_dir)
    print('---------------------------------------')
    print('Ensemble 3, aggregation results: ')
    print('Whole: ' + str(agg_3_whole))
    print('Core: ' + str(agg_3_core))
    print('Enhancing: ' + str(agg_3_enh))
    print('---------------------------------------')

    #Calculate average Dice ensemble scores using 4 modalities and voting ensemble mechanism - save final segmentation map
    voting_4_whole,voting_4_core,voting_4_enh = get_votes_4(file_names, t1_dir, t1c_dir, t2_dir, flair_dir, gt_dir, voting_4_out_dir)
    print('---------------------------------------')
    print('Ensemble 4, voting results: ')
    print('Whole: ' + str(voting_4_whole))
    print('Core: ' + str(voting_4_core))
    print('Enhancing: ' + str(voting_4_enh))
    print('---------------------------------------')

    #Calculate average Dice ensemble scores using 3 modalities and voting ensemble mechanism - save final segmentation map
    voting_3_whole,voting_3_core,voting_3_enh = get_votes_3(file_names,t1c_dir, t2_dir, flair_dir, gt_dir, voting_3_out_dir)
    print('---------------------------------------')
    print('Ensemble 3, voting results: ')
    print('Whole: ' + str(voting_3_whole))
    print('Core: ' + str(voting_3_core))
    print('Enhancing: ' + str(voting_3_enh))
    print('---------------------------------------')

if __name__ == '__main__':
    main()