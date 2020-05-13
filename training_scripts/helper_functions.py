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
from bxs633.model.helper_functions import *
from bxs633.model.unet import *
from bxs633.model.my_data_generator import *

def training_func(optim, n_epochs, output_dir, file_name, training_generator, val_generator, loss_func=dice_coefficient_loss_gen, eval_metric=dice_coefficient):
    #Checkpointing model
    checkpoint_file = output_dir + 'checkpoints/' + file_name + '.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, mode='min')

    #Saving data for visulisation
    tb_dir = output_dir + 'tensorflow_dir/' + file_name
    tb = TensorBoard(log_dir=tb_dir, histogram_freq=0, batch_size=1, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    #Saving all output to csv
    csv_name = output_dir + 'csv_logs/' + file_name
    csv_logger = CSVLogger(csv_name)

    #Add early stopping to stop model training when validation loss isn't improving
    es = EarlyStopping(monitor='val_loss',  min_delta=0, patience=100)

    #Decrease learning rate if validation loss doesn't decrease after 5 epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    
    callbacks_list = [checkpoint, tb, csv_logger, es, reduce_lr]

    model = unet_gen(optim)
    
    model.fit_generator(generator=training_generator, epochs=n_epochs, verbose=2, validation_data=val_generator, callbacks=callbacks_list)