import numpy as np
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import warnings
import tensorflow as tf

""" Generic Dice Coefficient 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

""" Multi-label Dice Coefficient 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coef_multilabel(y_true, y_pred, numLabels=5):
    dice=0
    for index in range(numLabels):
        dice += dice_coefficient(y_true[:,index,:,:,:], y_pred[:,index,:,:,:])
    return dice

""" Average Multi-label Dice Coefficient 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coef_multilabel_avg(y_true, y_pred, numLabels=5):
    dice=0
    for index in range(numLabels):
        dice += dice_coefficient(y_true[:,index,:,:,:], y_pred[:,index,:,:,:])
    return dice/4

""" Weighted Dice Coefficient 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coef_weighted(y_true, y_pred, numLabels=5, class_weights=[0.95459094, 0.0104403, 0.02643511, 0, 0.00853365]):
    dice=0
    for index in range(numLabels):
        dice += (dice_coefficient(y_true[:,index,:,:,:], y_pred[:,index,:,:,:]) * class_weights[index])
    return dice

""" Average weighted Dice Coefficient 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coef_weighted_avg(y_true, y_pred, numLabels=5, class_weights=[0.95459094, 0.0104403, 0.02643511, 0, 0.00853365]):
    dice=0
    for index in range(numLabels):
        dice += (dice_coefficient(y_true[:,index,:,:,:], y_pred[:,index,:,:,:]) * class_weights[index])
    return dice/4

""" Balanced class weights weighted Dice Coefficient 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coef_balanced(y_true, y_pred, numLabels=5, class_weights=[0.26189228, 23.94566934, 9.45712057, 0, 29.29579758]):
    dice=0
    for index in range(numLabels):
        dice += (dice_coefficient(y_true[:,index,:,:,:], y_pred[:,index,:,:,:]) * class_weights[index])
    return dice


""" Average Balanced class weights weighted Dice Coefficient 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coef_balanced_avg(y_true, y_pred, numLabels=5, class_weights=[0.26189228, 23.94566934, 9.45712057, 0, 29.29579758]):
    dice=0
    for index in range(numLabels):
        dice += (dice_coefficient(y_true[:,index,:,:,:], y_pred[:,index,:,:,:]) * class_weights[index])
    return dice/4


""" Soft Dice Loss 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coefficient_loss_gen(y_true, y_pred):
    return (1-dice_coefficient(y_true, y_pred))

""" Multi-label Dice Loss
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coefficient_multilable_loss(y_true, y_pred):
    return (1-dice_coef_multilabel(y_true, y_pred))

""" Average Multi-label Dice Loss 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coefficient_multilable_avg_loss(y_true, y_pred):
    return (1-dice_coef_multilabel_avg(y_true, y_pred))

""" Weighted Dice Loss 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coefficient_weighted_loss(y_true, y_pred):
    return (1-dice_coef_weighted(y_true, y_pred))

""" Average Weighted Dice Loss 
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coefficient_weighted_avg_loss(y_true, y_pred):
    return (1-dice_coef_weighted_avg(y_true, y_pred))

""" Balanced class weights weighted Dice Loss
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coefficient_balanced_loss(y_true, y_pred):
    return (1-dice_coef_balanced(y_true, y_pred))

""" Average Balanced class weights weighted Dice Loss
    :param y_true: ground truth volume
    :param y_pred: predicted volume
    :return: Dice Score """
def dice_coefficient_balanced_avg_loss(y_true, y_pred):
    return (1-dice_coef_balanced_avg(y_true, y_pred))

