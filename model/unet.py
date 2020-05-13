import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam, Adadelta, SGD, Nadam
from keras.layers.merge import concatenate
from keras.backend.tensorflow_backend import set_session
from keras import losses
from keras import metrics
import warnings
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.callbacks.tensorboard_v1 import TensorBoard
from my_data_generator import *
from bxs633.model.metrics import *

""" Create a convolution layer followed by batch normalisationa and ReLU activation
   :param filters: filter size i.e. 8
   :param input_layer: previous layer over which new operations are to be carried out
   :param conv_kernal: convolution kernal size
   :param padding: padding to add during convolution
   :param strides: stride length of convolution
   :return: resulting layer after all operations """
def conv_layer(filters, input_layer, conv_kernal, padding, strides):
    conv = Conv3D(filters, conv_kernal, padding=padding, strides=strides)(input_layer)
    norm = BatchNormalization(axis=1)(conv)
    activation = Activation('relu')(norm)
    return activation

""" Create a convolution block, made from 2 sets of convolution, batch normalisation and ReLU layers
   :param filters: filter size i.e. 8
   :param input_layer: previous layer over which new operations are to be carried out
   :param conv_kernal: convolution kernal size
   :param padding: padding to add during convolution
   :param strides: stride length of convolution
   :return: resulting layer after all operations """
def conv_block(filters, input_layer, conv_kernal, padding, strides):
    conv1 = conv_layer(filters, input_layer, conv_kernal, padding, strides)
    conv2 = conv_layer(filters, conv1, conv_kernal, padding, strides)
    return conv2


""" Build U-Net model
   :param optim: optimser
   :param loss_func: oss function
   :param eval_metric: evaluation metric
   :param base_filter: size of input
   :param depth: model depth
   :param final_activation: final activation function of network
   :return: model """
def unet_gen(optim=Adam(learning_rate=0.0001), loss_func=dice_coefficient_loss_gen, eval_metric=dice_coefficient, base_filter=8, patch_size=(1,128,128,128), depth=3, final_activation='sigmoid'):
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
    model.compile(optimizer=optim, loss=loss_func, metrics=[eval_metric])

    print(model.summary())
    return model




