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
   :param parent_dir: path to bias feild corrected data directory
   :param output_dir: path to co-registration output directory
   :return: list of tuples. Each tuple contains paths to t1,t1c,t2,flair files and output directory of co-registration """
def get_file_paths(parent_dir, out_dir):
    child_dirs = list(glob(parent_dir))

    mri_sets = []

    for child in child_dirs:
        t1_file = list(glob(child + '/*t1*'))
        t1c_file = list(glob(child + '/*t1c*'))
        t2_file = list(glob(child + '/*t2*'))
        flair_file = list(glob(child + '/*flair*'))

        mri_sets.append([t1_file[0], t1c_file[0], t2_file[0], flair_file[0], out_dir])

    print(len(mri_sets))
    return mri_sets

""" Co-register 4 modalities, normalise, crop and save result for a single patient
   :param paths: tuple containing paths to t1,t1c,t2,flair files of patient and output directory of co-registration
   :param output_dir: path to co-registration output directory """
def coreg_four_and_norm(paths):
    print('STARTING')
    start = time.time()
    t1_path, t1c_path, t2_path, flair_path, out_dir = paths
    
    patient_file = out_dir + (t1_path.split('/'))[-2] + '.npy'
    print('File Name: ' + patient_file)
    
    t1 = sitk.ReadImage(t1_path, sitk.sitkFloat32)
    t1c = sitk.ReadImage(t1c_path, sitk.sitkFloat32)
    t2 = sitk.ReadImage(t2_path, sitk.sitkFloat32)
    flair = sitk.ReadImage(flair_path, sitk.sitkFloat32)
    
    #Apply co-registration
    co_reg_one, transformation_one = apply_registration(t2, t1)
    co_reg_two, transformation_two = apply_registration(co_reg_one, flair)
    co_reg_three, transformation_three = apply_registration(co_reg_two, t1c)
    #Apply normalisation to image
    coreg_normalised = normalise(co_reg_three)
    

    #Change image to numpy.ndarray format. This removes preprocessing from data loader.
    #Numpy format allows data loading processed to be a lot faster - reducing training time by quite a lot.
    coreg_normalised = sitk.GetArrayFromImage(coreg_normalised)
    coreg_cropped = center_crop(coreg_normalised)
    coreg_cropped = coreg_cropped.reshape([1, coreg_cropped.shape[0], coreg_cropped.shape[1], coreg_cropped.shape[2]])
    print(coreg_cropped.shape)
    print(type(coreg_cropped))
    #Save final image in numpy format
    np.save(patient_file, coreg_cropped)
    end = time.time()
    print('Time taken: ' + str(end-start))

""" Co-register 3 modalities, normalise, crop and save result for a single patient
   :param paths: tuple containing paths to t1,t1c,t2,flair files of patient and output directory of co-registration
   :param output_dir: path to co-registration output directory """
def coreg_three_and_norm(paths):
    print('STARTING')
    start = time.time()
    t1_path, t1c_path, t2_path, flair_path, out_dir = paths
    
    patient_file = out_dir + (t1_path.split('/'))[-2] + '.npy'
    print('File Name: ' + patient_file)
    
    t1c = sitk.ReadImage(t1c_path, sitk.sitkFloat32)
    t2 = sitk.ReadImage(t2_path, sitk.sitkFloat32)
    flair = sitk.ReadImage(flair_path, sitk.sitkFloat32)

    co_reg_one, transformation_two = apply_registration(t2, flair)
    co_reg_two, transformation_three = apply_registration(co_reg_one, t1c)
    coreg_normalised = normalise(co_reg_two)
    
    #Change image to numpy.ndarray format. This removes preprocessing from data loader.
    #Numpy format allows data loading processed to be a lot faster - reducing training time by quite a lot.
    coreg_normalised = sitk.GetArrayFromImage(coreg_normalised)
    coreg_cropped = center_crop(coreg_normalised)
    coreg_cropped = coreg_cropped.reshape([1, coreg_cropped.shape[0], coreg_cropped.shape[1], coreg_cropped.shape[2]])
    print(coreg_cropped.shape)
    print(type(coreg_cropped))
    #Save final image in numpy format
    np.save(patient_file, coreg_cropped)
    end = time.time()
    print('Time taken: ' + str(end-start))

"""Co-register, normalise, crop and save over whole data set in parallel
   :param num_of_mods: number of modalities to be co-registered. 3 or 4
   :param parent_dir: path to bias feild corrected data directory
   :param output_dir: path to co-registration output directory """
def main():
    num_of_mods = int(sys.argv[1])
    parent_dir = sys.argv[2]
    out_dir = sys.argv[3]

    mri_dirs = get_file_paths(parent_dir, out_dir)
    p = Pool(3)

    if(num_of_mods == 3):
        p.map(coreg_three_and_norm, mri_dirs)
    elif(num_of_mods == 4):
        p.map(coreg_four_and_norm, mri_dirs)
    else:
        print("Modality value can only be 3 or 4")

if __name__ == '__main__':
    main()
