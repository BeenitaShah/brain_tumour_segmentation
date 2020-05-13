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
from keras.optimizers import Adam, Nadam, Adadelta
from keras.layers.merge import concatenate
from keras.backend.tensorflow_backend import set_session
import warnings
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import statistics
import sys
import math
from bxs633.model.my_data_generator import *
from bxs633.model.metrics import *
from bxs633.model.unet import *

""" Visualise slices of a 3D image
   :param data: image volume
   :param slices: list of indices of slices
   :param cmap: colour map of lables to colour """
def show_slices(data, slice_nums, cmap=None):
    fig = plt.figure(figsize=(15,10))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        plt.axis('off')

""" Create model from saved weights
   :param model_weight_file: file path to trained model
   :param loss_func: loss function model was trained with
   :param eval_metric: evaluation metric model was trained with
   :param base_filter: inital filter size model was trained with
   :param patch_size: input patch size model was trained with
   :param depth: depth of trained model
   :param final_activation: final activation model was trained with 
   :returns model """
def unet_inference(model_weight_file, loss_func=dice_coefficient_loss_gen, eval_metric=dice_coefficient, base_filter=8, patch_size=(1,128,128,128), depth=3, final_activation='sigmoid'):
    base_filter = base_filter
    inputs = Input(patch_size)

    conv_kernal = (3,3,3)
    padding = 'same'
    strides = (1,1,1)
    pool_size = (2,2,2)
    upsample_kernal = (2,2,2)

    #Downsampling block
    layers_downsample = []
    current_layer = inputs
    current_filter_size = base_filter*2
    for layer in range(depth+1):
        layer_conv = conv_block(current_filter_size, current_layer, conv_kernal, padding, strides)

        if(layer < depth):
            current_layer = MaxPooling3D(pool_size=pool_size)(layer_conv)
            layers_downsample.append([layer_conv, current_layer])
            current_filter_size = current_filter_size*2
        else:
            current_layer = layer_conv
            layers_downsample.append([layer_conv])
    
    #Upsampling block
    for layer in range(depth-1,-1,-1):
        deconv = Deconvolution3D(filters=current_filter_size, kernel_size=upsample_kernal, strides=upsample_kernal)(current_layer)
        concat = concatenate([deconv, layers_downsample[layer][0]], axis=1)
        
        current_filter_size = current_filter_size // 2
        current_layer = conv_block(current_filter_size, concat, conv_kernal, padding, strides)
    
    final_layer = Conv3D(5, (1,1,1))(current_layer)
    act = Activation(final_activation)(final_layer)

    model = Model(inputs=inputs, outputs=act)
    model.load_weights(model_weight_file)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss=loss_func, metrics=[eval_metric])

    #print(model.summary())
    return model

""" Change predictions into segmentation map
   :param data: prediction volume
   :returns segmentation map """
def get_labels(data):
    return np.argmax(data, axis=0)

""" Get whole tumour mask from prediction
   :param data: prediction volume
   :returns whole tumour mask """
def get_whole_tumor_mask(data):
    return data > 0

""" Get tumour core mask from prediction
   :param data: prediction volume
   :returns tumour core mask """
def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)

""" Get enhancing tumour mask from prediction
   :param data: prediction volume
   :returns enhancing tumour mask """
def get_enhancing_tumor_mask(data):
    return data == 4

""" Calculate Dice coefficient over label masks
   :param truth: ground truth label mask 
   :param prediction: prediction label mask 
   :returns dice score """
def dice_coefficient_over_lables(truth, prediction):
    lable_vals = 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))
    if(math.isnan(lable_vals)):
        return 0
    else:
        return lable_vals

""" Centeral crop volume to centeral 128x128x128 pixels
   :param img: image to be cropped
   :returns cropped volume """
def center_crop(image):
        slices = 128
        length = 128
        height = 128
        
        slice_from = (image.shape[0] - slices) // 2
        length_from = (image.shape[1] - length) //2
        height_from = (image.shape[2] - height) // 2
        
        slice_to = slice_from + slices
        length_to = length_from + length
        height_to = height_from + height
        
        return image[slice_from:slice_to, length_from:length_to, height_from:height_to]

""" Calculate segmentation map over 4 modalities using aggregation ensemble
   :param file_names: name of file names (only) in the test set
   :param t1_dir: T1 data directory
   :param t1c_dir: T1C data directory
   :param t2_dir: T2 data directory
   :param flair_dir: Flair data directory
   :param out_dir: directory path for where predictions should be saved
   :returns average dice score for each label group over test set """
def get_agg_4(file_names, t1_dir, t1c_dir, t2_dir, flair_dir, gt_dir, out_dir):
    agg_dice_whole = []
    agg_dice_core = []
    agg_dice_enhancing = []

    for file in file_names:
        gt = np.load(gt_dir + file)
        gt = gt[0,:,:,:]
        gt = center_crop(gt)
        
        t1 = np.load(t1_dir + file)
        flair = np.load(flair_dir + file)
        t1c = np.load(t1c_dir + file)
        t2 = np.load(t2_dir + file)

        agg_pred = np.add(t1, flair)
        agg_pred = np.add(agg_pred, t1c)
        agg_pred = np.add(agg_pred, t2)
        agg_pred = np.true_divide(agg_pred, 4)
        agg_pred = get_labels(agg_pred)
        file_name = out_dir + file
        np.save(file_name, agg_pred)
        
        gt_whole_tumour = get_whole_tumor_mask(gt)
        gt_core = get_tumor_core_mask(gt)
        gt_enhancing = get_enhancing_tumor_mask(gt)

        pred_whole_tumour = get_whole_tumor_mask(agg_pred)
        pred_core = get_tumor_core_mask(agg_pred)
        pred_enhancing = get_enhancing_tumor_mask(agg_pred)

        current_whole = dice_coefficient_over_lables(gt_whole_tumour, pred_whole_tumour)
        current_core = dice_coefficient_over_lables(gt_core, pred_core)
        current_enhancing = dice_coefficient_over_lables(gt_enhancing, pred_enhancing)

        agg_dice_whole.append(current_whole)
        agg_dice_core.append(current_core)
        agg_dice_enhancing.append(current_enhancing)

    #Calculating Averages
    agg_avg_whole = sum(agg_dice_whole) / len(agg_dice_whole)
    agg_avg_core = sum(agg_dice_core) / len(agg_dice_core)
    agg_avg_enhancing = sum(agg_dice_enhancing) / len(agg_dice_enhancing)
    # print(agg_avg_whole)
    # print(agg_avg_core)
    # print(agg_avg_enhancing)
    return agg_avg_whole, agg_avg_core, agg_avg_enhancing

""" Calculate segmentation map over 3 modalities using aggregation ensemble
   :param file_names: name of file names (only) in the test set
   :param t1_dir: T1 data directory
   :param t1c_dir: T1C data directory
   :param t2_dir: T2 data directory
   :param flair_dir: Flair data directory
   :param out_dir: directory path for where predictions should be saved
   :returns average dice score for each label group over test set """
def get_agg_3(file_names, t1c_dir, t2_dir, flair_dir, gt_dir, out_dir):
    agg_dice_whole = []
    agg_dice_core = []
    agg_dice_enhancing = []

    for file in file_names:
        gt = np.load(gt_dir + file)
        gt = gt[0,:,:,:]
        gt = center_crop(gt)
        
        flair = np.load(flair_dir + file)
        t1c = np.load(t1c_dir + file)
        t2 = np.load(t2_dir + file)

        agg_pred = np.add(flair, t1c)
        agg_pred = np.add(agg_pred, t2)
        agg_pred = np.true_divide(agg_pred, 3)
        agg_pred = get_labels(agg_pred)
        file_name = out_dir + file
        np.save(file_name, agg_pred)
        
        gt_whole_tumour = get_whole_tumor_mask(gt)
        gt_core = get_tumor_core_mask(gt)
        gt_enhancing = get_enhancing_tumor_mask(gt)

        pred_whole_tumour = get_whole_tumor_mask(agg_pred)
        pred_core = get_tumor_core_mask(agg_pred)
        pred_enhancing = get_enhancing_tumor_mask(agg_pred)

        current_whole = dice_coefficient_over_lables(gt_whole_tumour, pred_whole_tumour)
        current_core = dice_coefficient_over_lables(gt_core, pred_core)
        current_enhancing = dice_coefficient_over_lables(gt_enhancing, pred_enhancing)

        agg_dice_whole.append(current_whole)
        agg_dice_core.append(current_core)
        agg_dice_enhancing.append(current_enhancing)

    #Calculating Averages
    agg_avg_whole = sum(agg_dice_whole) / len(agg_dice_whole)
    agg_avg_core = sum(agg_dice_core) / len(agg_dice_core)
    agg_avg_enhancing = sum(agg_dice_enhancing) / len(agg_dice_enhancing)
    # print(agg_avg_whole)
    # print(agg_avg_core)
    # print(agg_avg_enhancing)
    return avg_whole,avg_core,avg_enhancing

""" Calculate segmentation map from given list
   :param preds: predicted class from each modality
   :returns class which appears the most """
def find_max_mode(preds):
    list_table = statistics._counts(preds)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(preds)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode

""" Calculate segmentation map over 4 modalities using voting ensemble for a single patient
   :param t1: T1 volume
   :param t1c: T1C volume
   :param t2: T2 volume
   :param flair: Flair volume
   :returns prediction segmentation map """
def max_voting_4(t1,t1c,t2,flair):
    t1 = get_labels(t1)
    t1c = get_labels(t1c)
    t2 = get_labels(t2)
    flair = get_labels(flair)
    pred = np.zeros(shape=(128,128,128))
    
    for (x,y,z), value in np.ndenumerate(t1):
        pred[x,y,z] = find_max_mode([t1[x,y,z], t1c[x,y,z], t2[x,y,z], flair[x,y,z]])
    
    print(pred.shape)
    print(np.unique(pred))
    return pred

""" Calculate segmentation map over 3 modalities using voting ensemble for a single patient
   :param t1c: T1C volume
   :param t2: T2 volume
   :param flair: Flair volume
   :returns prediction segmentation map """
def max_voting_3(t1c,t2,flair):
    t1c = get_labels(t1c)
    t2 = get_labels(t2)
    flair = get_labels(flair)
    pred = np.zeros(shape=(128,128,128))
    
    for (x,y,z), value in np.ndenumerate(t1c):
        pred[x,y,z] = find_max_mode([t1c[x,y,z], t2[x,y,z], flair[x,y,z]])
    
    print(pred.shape)
    print(np.unique(pred))
    return pred

""" Calculate segmentation map over 4 modalities using voting ensemble
   :param file_names: name of file names (only) in the test set
   :param t1_dir: T1 data directory
   :param t1c_dir: T1C data directory
   :param t2_dir: T2 data directory
   :param flair_dir: Flair data directory
   :param out_dir: directory path for where predictions should be saved
   :returns average dice score for each label group over test set """
def get_votes_4(file_names, t1_dir, t1c_dir, t2_dir, flair_dir, gt_dir, out_dir):
    dice_whole = []
    dice_core = []
    dice_enhancing = []

    for file in file_names:
        gt = np.load(gt_dir + file)
        gt = gt[0,:,:,:]
        gt = center_crop(gt)
        
        t1 = np.load(t1_dir + file)
        flair = np.load(flair_dir + file)
        t1c = np.load(t1c_dir + file)
        t2 = np.load(t2_dir + file)
        
        pred = max_voting_4(t1,t1c,t2,flair)
        file_name = out_dir + file
        np.save(file_name, pred)
        
        gt_whole_tumour = get_whole_tumor_mask(gt)
        gt_core = get_tumor_core_mask(gt)
        gt_enhancing = get_enhancing_tumor_mask(gt)
    
        pred_whole_tumour = get_whole_tumor_mask(pred)
        pred_core = get_tumor_core_mask(pred)
        pred_enhancing = get_enhancing_tumor_mask(pred)

        current_whole = dice_coefficient_over_lables(gt_whole_tumour, pred_whole_tumour)
        current_core = dice_coefficient_over_lables(gt_core, pred_core)
        current_enhancing = dice_coefficient_over_lables(gt_enhancing, pred_enhancing)
        
        dice_whole.append(current_whole)
        dice_core.append(current_core)
        dice_enhancing.append(current_enhancing)

    avg_whole = sum(dice_whole) / len(dice_whole)
    avg_core = sum(dice_core) / len(dice_core)
    avg_enhancing = sum(dice_enhancing) / len(dice_enhancing)

    # print('Whole: ' + str(avg_whole))
    # print('Core: ' + str(avg_core))
    # print('Enhancing: ' + str(avg_enhancing))
    return avg_whole,avg_core,avg_enhancing


""" Calculate segmentation map over 3 modalities using voting ensemble
   :param file_names: name of file names (only) in the test set
   :param t1_dir: T1 data directory
   :param t1c_dir: T1C data directory
   :param t2_dir: T2 data directory
   :param flair_dir: Flair data directory
   :param out_dir: directory path for where predictions should be saved
   :returns average dice score for each label group over test set """
def get_votes_3(file_names, t1c_dir, t2_dir, flair_dir, gt_dir, out_dir):
    dice_whole = []
    dice_core = []
    dice_enhancing = []

    for file in file_names:
        gt = np.load(gt_dir + file)
        gt = gt[0,:,:,:]
        gt = center_crop(gt)
        
        flair = np.load(flair_dir + file)
        t1c = np.load(t1c_dir + file)
        t2 = np.load(t2_dir + file)
        
        pred = max_voting_3(t1c,t2,flair)
        file_name = out_dir + file
        np.save(file_name, pred)
        
        gt_whole_tumour = get_whole_tumor_mask(gt)
        gt_core = get_tumor_core_mask(gt)
        gt_enhancing = get_enhancing_tumor_mask(gt)
    
        pred_whole_tumour = get_whole_tumor_mask(pred)
        pred_core = get_tumor_core_mask(pred)
        pred_enhancing = get_enhancing_tumor_mask(pred)

        current_whole = dice_coefficient_over_lables(gt_whole_tumour, pred_whole_tumour)
        current_core = dice_coefficient_over_lables(gt_core, pred_core)
        current_enhancing = dice_coefficient_over_lables(gt_enhancing, pred_enhancing)
        
        dice_whole.append(current_whole)
        dice_core.append(current_core)
        dice_enhancing.append(current_enhancing)

    avg_whole = sum(dice_whole) / len(dice_whole)
    avg_core = sum(dice_core) / len(dice_core)
    avg_enhancing = sum(dice_enhancing) / len(dice_enhancing)

    # print('Whole: ' + str(avg_whole))
    # print('Core: ' + str(avg_core))
    # print('Enhancing: ' + str(avg_enhancing))
    return avg_whole,avg_core,avg_enhancing

""" Get list of file names in a directory
   :param data_path: directory path
   :returns list of file names in the directory """
def get_data(data_path):
    direct = list(glob(data_path + '*'))
    three_quarters = round(len(direct)*0.75)
    test_data = direct[three_quarters:]
    print(test_data[0])
    test_data =[(i.split('/'))[-1] for i in test_data]
    return test_data

""" Run inference over trained model
   :param loss_function: loss function model was trained with
   :param eval_metric_param: evaluation metric model was trained with
   :param model_weight_file: file path to trained model
   :param output_dir: directory path to save prediction
   :param test_generator: Dataloader for test set
   :param gt_dir: ground truth directory path
   :param test_data: list of file names in test data set
   :returns average dice scores for 3 label groups calculated over whole test set """
def run_inference(loss_function, eval_metric_param, model_weight_file, output_dir, test_generator, gt_dir, test_data):
    print('Running Model Inference for: ' + model_weight_file)
    model = unet_inference(model_weight_file, loss_func=loss_function, eval_metric=eval_metric_param)
    print('Calling Prediction')
    preds = model.predict_generator(test_generator)

    dice_whole = []
    dice_core = []
    dice_enhancing = []

    print('Calculating averages')
    for i,pred in enumerate(preds):
        gt = np.load(gt_dir + test_data[i])
        gt = gt[0,:,:,:]

        output_path = output_dir + test_data[i]
        np.save(output_path, pred)

        pred = get_labels(pred)
        
        gt_whole_tumour = get_whole_tumor_mask(gt)
        gt_core = get_tumor_core_mask(gt)
        gt_enhancing = get_enhancing_tumor_mask(gt)

        pred_whole_tumour = get_whole_tumor_mask(pred)
        pred_core = get_tumor_core_mask(pred)
        pred_enhancing = get_enhancing_tumor_mask(pred)

        current_whole = dice_coefficient_over_lables(gt_whole_tumour, pred_whole_tumour)
        current_core = dice_coefficient_over_lables(gt_core, pred_core)
        current_enhancing = dice_coefficient_over_lables(gt_enhancing, pred_enhancing)
        
        dice_whole.append(current_whole)
        dice_core.append(current_core)
        dice_enhancing.append(current_enhancing)
    
    avg_whole = sum(dice_whole) / len(dice_whole)
    avg_core = sum(dice_core) / len(dice_core)
    avg_enhancing = sum(dice_enhancing) / len(dice_enhancing)

    print('Whole: ' + str(avg_whole))
    print('Core: ' + str(avg_core))
    print('Enhancing: ' + str(avg_enhancing))

    return avg_whole, avg_core, avg_enhancing





