import numpy as np
from tensorflow.keras.utils import Sequence
from medpy.io import load
from medpy.io import header
from glob import glob
import SimpleITK as sitk
import sys
import os
from multiprocessing import Pool
import gc
from shutil import copyfile
import time
from all_preprocessing_methods import *

""" Get list contatining tuples of t1,t1c,t2,flair file paths and t1,t1c,t2,flair output directory paths for each patient
   :param parent_dir: path to bias feild corrected data directory
   :param t1_out: path to t1 normalised data output directory
   :param t1c_out: path to t1c normalised data output directory
   :param t2_out: path to t2 normalised data output directory
   :param flair_out: path to flair normalised data output directory
   :return: list of tuples. Each tuple contains paths to t1,t1c,t2,flair files and output directories for normalsied t1,t1c,t2,flair files """
def get_file_paths(parent_dir, t1_out, t1c_out, t2_out, flair_out):
    all_test = list(glob(parent_dir))
    mri_sets = []

    for child in all_test:
        t1_file = list(glob(child + '/*t1*'))
        t1c_file = list(glob(child + '/*t1c*'))
        t2_file = list(glob(child + '/*t2*'))
        flair_file = list(glob(child + '/*flair*'))

        mri_sets.append([t1_file[0], t1c_file[0], t2_file[0], flair_file[0], t1_out, t1c_out, t2_out, flair_out])

    print(len(mri_sets))
    return mri_sets

""" Normalise, crop and save image
   :param file_path: file path of image to be normalised
   :param output_path: file output path """
def normalise_modality(file_path, output_path):
    # print(file_path)
    # print(output_path)
    normalised = only_normalise(file_path)
    cropped = center_crop(normalised)
    cropped = cropped.reshape([1, cropped.shape[0], cropped.shape[1], cropped.shape[2]])
    # print(cropped.shape)
    # print(type(cropped))
    np.save(output_path, cropped)

""" Normalise, crop and save image for all modalities of a single patient
   :param paths: tuple containing t1, t1c,t2,flair file paths and output directories for normalised t1,t1c,t2,flair files """
def apply_mod_normalisation(paths):
    t1_path, t1c_path, t2_path, flair_path, t1_out_dir, t1c_out_dir, t2_out_dir, flair_out_dir = paths
    
    patient_file= (t1_path.split('/'))[-2] + '.npy'

    t1_out = t1_out_dir + patient_file
    t1c_out = t1c_out_dir + patient_file
    t2_out = t2_out_dir + patient_file
    flair_out = flair_out_dir + patient_file

    # print('STARTING')
    start = time.time()

    normalise_modality(t1_path,t1_out)
    normalise_modality(t1c_path, t1c_out)
    normalise_modality(t2_path, t2_out)
    normalise_modality(flair_path,flair_out)
    
    end = time.time()
    print('Time taken: ' + str(end-start))

""" Normalise all modalities for all patients in parallel
   :param parent_dir: path to bias feild corrected data directory
   :param t1_out: path to t1 normalised data output directory
   :param t1c_out: path to t1c normalised data output directory
   :param t2_out: path to t2 normalised data output directory
   :param flair_out: path to flair normalised data output directory """
def main():
    parent = sys.argv[1] + '*'
    t1_out = sys.argv[2]
    t1c_out = sys.argv[3]
    t2_out = sys.argv[4]
    flair_out = sys.argv[5]

    mri_dirs = get_file_paths(parent, t1_out, t1c_out, t2_out, flair_out)
    p = Pool(3)
    p.map(apply_mod_normalisation, mri_dirs)

if __name__ == '__main__':
    main()
