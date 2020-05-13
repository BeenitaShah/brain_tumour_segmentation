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

""" Get list contatining tuples of t1,t1c,t2,flair file paths and bias field output directory path for each patient
   :param parent_hgg: path to HGG data directiory
   :param parent_lgg: path to LGG data directiory
   :param output_directory: path to bias field correction parent output directory
   :return: list of tuples. Each tuple contains paths to t1,t1c,t2,flair files and output directory of bias field correction """
def get_file_paths(parent_hgg, parent_lgg, out_dir):
    child_HGG = list(glob(parent_hgg))
    child_LGG = list(glob(parent_lgg))

    mri_sets = []

    all_test = child_HGG + child_LGG

    for child in all_test:
        t1_file = list(glob(child + '/*t1*'))
        t1c_file = list(glob(child + '/*t1c*'))
        t2_file = list(glob(child + '/*t2*'))
        flair_file = list(glob(child + '/*flair*'))

        mri_sets.append([t1_file[0], t1c_file[0], t2_file[0], flair_file[0], out_dir])

    # print(len(mri_sets))
    return mri_sets

""" Apply bias feild correction over t1,t1c,t2 and flair files for a single patient and save to patient sub-directory under given parent directory
   :param paths: tuple - t1,t1c,t2,flair file paths and bias field correction output directoy """
def apply_bias_correction(paths):
    t1_path, t1c_path, t2_path, flair_path, out_dir = paths
    # print(t1_path)
    
    if('HGG' in t1_path):
        patient_dir = (t1_path.split('/'))[-2] +'_HGG/'
    else:
        patient_dir = (t1_path.split('/'))[-2] +'_LGG/'
    
    patient_dir = out_dir + patient_dir
    print(patient_dir)
    if not os.path.exists(patient_dir):
        #print('making dir')
        os.makedirs(patient_dir)

    t1_out = patient_dir + 't1.nii.gz'
    t1c_out = patient_dir + 't1c.nii.gz'
    t2_out = patient_dir + 't2.nii.gz'
    flair_out = patient_dir + 'flair.nii.gz'

    # print('STARTING')
    start = time.time()
    
    t1_corrected = bias_field_correction(t1_path)
    sitk.WriteImage(t1_corrected, t1_out)
    # print('T1 Done')
    t1c_corrected = bias_field_correction(t1c_path)
    sitk.WriteImage(t1c_corrected, t1c_out)
    # print('T1c Done')
    t2_corrected = bias_field_correction(t2_path)
    sitk.WriteImage(t2_corrected, t2_out)
    # print('T2 Done')
    flair_corrected = bias_field_correction(flair_path)
    sitk.WriteImage(flair_corrected, flair_out)
    # print('Flair Done')
    
    end = time.time()
    print('Time taken: ' + str(end-start))

""" Call Bias field correction over all files in parallel
   :param parent_hgg: path to HGG data directiory
   :param parent_lgg: path to LGG data directiory
   :param output_directory: path to bias field correction parent output directory """
def main():
    parent_hgg = sys.argv[1]
    parent_lgg = sys.argv[2]
    out_dir = sys.argv[3]

    mri_dirs = get_file_paths(parent_hgg, parent_lgg, out_dir)
    p = Pool(3)
    p.map(apply_bias_correction, mri_dirs)

if __name__ == '__main__':
    main()
