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

""" Crop volume to centeral 128x128x128 pixels
   :param image: pre-loaded image object/volume - 3 dimensional
   :return: cropped image """
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

""" Load grayscale image
        :param image_path: path to image file
        :return: loaded image """
def _load_grayscale_image(image_path):
        mri_image_data, mri_image_header = load(image_path)
        # print(mri_image_data.shape)
        mri_image_data = center_crop(mri_image_data)
        mri_image_data = mri_image_data.reshape([1, mri_image_data.shape[0],mri_image_data.shape[1], mri_image_data.shape[2]])
        # print(mri_image_data.shape)
        return mri_image_data

""" Normalise pixel values
   :param image: pre-loaded image object/volume - 3 dimensional
   :return: normalised image """
def normalise(image):
    normalizeFilter = sitk.NormalizeImageFilter()
    outputIm = normalizeFilter.Execute(image)
    return outputIm

""" Normalise pixel values
   :param file: = path to image to be normalised
   :return: normalised image as numpy array """
def only_normalise(file):
    image = sitk.ReadImage(file, sitk.sitkFloat32)
    normalizeFilter = sitk.NormalizeImageFilter()
    outputIm = normalizeFilter.Execute(image)
    outputIm = sitk.GetArrayFromImage(outputIm)
    return outputIm

""" Change 3 dimensional image to 4 dimensional volume
   :param image: pre-loaded image object/volume - 3 dimensional
   :return: 4D image with volume (1,image[0],image[1],image[2]) """
def reshape(image):
    image = image.reshape([1, image.shape[0],image.shape[1], image.shape[2]])
    return image

""" Change 3 dimensional image to 4 dimensional volume
   :param paths: a tuple of size 5 containing paths to: t1,t1c,t2,flair files and patient output directory where all normalised images are to be saved. """
def apply_normalisation(paths):
    t1_path, t1c_path, t2_path, flair_path, out_dir = paths

    if('HGG' in t1_path):
        extension = 'hgg'
    else:
        extension = 'lgg'

    patient_dir = out_dir + (t1_path.split('/'))[-2] + '_' + extension + '/'
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)
    
    if(not(os.path.isfile(patient_dir))):
        t1_out = patient_dir + 't1.npy'
        t1c_out = patient_dir + 't1c.npy'
        t2_out = patient_dir + 't2.npy'
        flair_out = patient_dir + 'flair.npy'

        t1_normalised = only_normalise(t1_path)
        t1_normalised = center_crop(t1_normalised)
        t1_normalised = reshape(t1_normalised)
        # print(t1_normalised.shape)
        # print('t1 done')
        t1c_normalised = only_normalise(t1c_path)
        t1c_normalised = center_crop(t1c_normalised)
        t1c_normalised = reshape(t1c_normalised)
        # print(t1c_normalised.shape)
        # print('t1c done')
        t2_normalised = only_normalise(t2_path)
        t2_normalised = center_crop(t2_normalised)
        t2_normalised = reshape(t2_normalised)
        # print(t2_normalised.shape)
        # print('t2 done')
        flair_normalised = only_normalise(flair_path)
        flair_normalised = center_crop(flair_normalised)
        flair_normalised = reshape(flair_normalised)
        # print(flair_normalised.shape)
        # print('flair done')

        np.save(t1_out, t1_normalised)
        np.save(t1c_out, t2_normalised)
        np.save(t2_out, t2_normalised)
        np.save(flair_out, flair_normalised)

        # print('Another one done...')
        # print('----------------')
    else:
        print('File already processed')

""" Change ground truth volumes from .nii.gz/mha formats to numpy arrays and save in that format
   :param paths: a tuple of size 2: path to ground truth and parent directory where new ground truths are to be saved """
def truth_arrays_to_numpy(paths):
    gt_path, out_dir = paths

    if('HGG' in gt_path):
        extension = 'hgg'
    else:
        extension = 'lgg'

    patient_dir = out_dir + (gt_path.split('/'))[-2] + '_' + extension + '/'
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)
    
    gt_out = patient_dir + 'gt.npy'
    # print(gt_out)
    gt = sitk.ReadImage(gt_path, sitk.sitkFloat32)
    gt = sitk.GetArrayFromImage(gt)
    gt = center_crop(gt)
    gt = reshape(gt)
    np.save(gt_out, gt)

""" Bias Field correct an image
   :param file: path to image file
   :return: bias feild corrected image """
def bias_field_correction(file):
    image = sitk.ReadImage(file, sitk.sitkFloat32)
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    iterations = 5
    corrector.SetMaximumNumberOfIterations([iterations] *numberFittingLevels )
    output = corrector.Execute(image, maskImage)
    return output

""" Co-register two images
   :param fixed_image: pre-loaded fixed image. Basis image for co-registration
   :param moving_image: pre-loaded moving image.
   :return: (re_sampled image, transformation parameters) """
def apply_registration(moving_image, fixed_image):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))

    # print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    return (moving_resampled,final_transform)

""" Change file name convention 
   :param directory_path: path to directory of file to be corrected """
def rename_training(directory_path):
    all_file_paths = list(glob(directory_path))
    print(len(all_file_paths))
    for name in all_file_paths:
        print(name)
        path = name.split('/')[:-1]
        path = '/'.join(path)
        print(path)
        new_name = name.split('/')[-1]
        if('hgg' in name):
            new_name = new_name.split('hgg')[0] + 'HGG.npy'
        else:
            new_name = new_name.split('lgg')[0] + 'LGG.npy'
        new_name = path + '/' + new_name
        print('New Name: ' + new_name)
        os.rename(name, new_name)