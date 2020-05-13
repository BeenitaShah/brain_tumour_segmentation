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
from scipy.stats import t
from scipy import stats
import warnings
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from helper_functions import *
import sys  
from bxs633.model.my_data_generator import *
from bxs633.evaluation.helper_functions import *

""" Calculate whole, core and enhancing tumour for each prediction over test
   :param file_names: list of file names in test set
   :param file_path: path to prediction data directory
   :param gt_dir: ground truth directory path
   :param is_coreg: boolean to specify is predictions are from co-reg data trained model
   :returns 3 lists of Dice scores for, one for each tumour type """
def get_preds(file_names, file_path, gt_dir, is_coreg):
    dice_whole = []
    dice_core = []
    dice_enhancing = []

    for file in file_names:
        gt = np.load(gt_dir + file)
        gt = gt[0,:,:,:]
        gt = center_crop(gt)
        
        pred_mask = np.load(file_path + file)
        if(is_coreg == 1):
            pred_mask = get_labels(pred_mask)


        gt_whole_tumour = get_whole_tumor_mask(gt)
        gt_core = get_tumor_core_mask(gt)
        gt_enhancing = get_enhancing_tumor_mask(gt)

        pred_whole_tumour = get_whole_tumor_mask(pred_mask)
        pred_core = get_tumor_core_mask(pred_mask)
        pred_enhancing = get_enhancing_tumor_mask(pred_mask)

        current_whole = dice_coefficient_over_lables(gt_whole_tumour, pred_whole_tumour)
        current_core = dice_coefficient_over_lables(gt_core, pred_core)
        current_enhancing = dice_coefficient_over_lables(gt_enhancing, pred_enhancing)

        dice_whole.append(current_whole)
        dice_core.append(current_core)
        dice_enhancing.append(current_enhancing)

    # Calculating Averages
    # avg_whole = sum(dice_whole) / len(dice_whole)
    # avg_core = sum(dice_core) / len(dice_core)
    # avg_enhancing = sum(dice_enhancing) / len(dice_enhancing)
    # print(avg_whole)
    # print(avg_core)
    # print(avg_enhancing)

    return dice_whole, dice_core, dice_enhancing

""" T-Test over different models
   :param pred_dir_one: path to prediction data directory for first model
   :param model_one_coreg: 1 if model is co-reg model else 0
   :param pred_dir_two: path to prediction data directory for second model
   :param model_two_coreg: 1 if model is co-reg model else 0
   :param gt_dir: ground truth directory path
   :returns t-values and p-values of T-test """
def main():
    pred_dir_one = sys.argv[1]
    model_one_coreg = int(sys.argv[2])
    pred_dir_two = sys.argv[3]
    model_two_coreg = int(sys.argv[4])
    gt_dir = sys.argv[5]

    #set of files in both directories
    model_one = list(glob(pred_dir_one + "*"))
    model_one_files = [i.split('/')[-1] for i in model_one]

    model_two = list(glob(pred_dir_two + "*"))
    model_two_files = [i.split('/')[-1] for i in model_two]

    file_names = list(set(model_one_files) &  set(model_two_files))
    print(len(file_names))

    print('T Test')
    one_whole, one_core, one_enh = get_preds(file_names, pred_dir_one, gt_dir, model_one_coreg)
    two_whole, two_core, two_enh = get_preds(file_names, pred_dir_two, gt_dir, model_two_coreg)

    whole = stats.ttest_ind(one_whole, two_whole, equal_var = False)
    print(whole)
    core = stats.ttest_ind(one_core, two_core, equal_var = False)
    print(core)
    enh = stats.ttest_ind(one_enh, two_enh, equal_var = False)
    print(enh)
    print('-----------------------------------')


if __name__ == '__main__':
    main()





