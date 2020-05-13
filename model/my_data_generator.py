import numpy as np
from tensorflow.keras.utils import Sequence
from medpy.io import load
from medpy.io import header
from glob import glob
from tensorflow.keras.utils import to_categorical

""" Dataloader """
class DataGenerator(Sequence):
    """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location (ground truth location)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of classes in output mask
        :param shuffle: True to shuffle label indexes after every epoch
        """
    def __init__(self, list_IDs, labels,image_path, mask_path,
                slices=128, height = 128, length = 128, to_fit=True, batch_size=1,
                n_channels=1, n_classes=5, shuffle=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.image_path = image_path
        self.mask_path =mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = (slices, height, length)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.slices = slices
        self.height = height
        self.length = length
        self.on_epoch_end()
        self.class_lables = [0,1,2,3,4]


    """ Returns number of batches per epoch """
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    """ Returns X and Y when training, only Y during inference 
        :param index: index of data point relative to whole data set
        :return: list of index to be loaded """
    def __getitem__(self, index):
        # Generate indexes of the batch 
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    # Shuffles order of data after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    """Generates data containing batch_size images
       :param list_IDs_temp: list of label ids to load
       :return: batch of images """
    def _generate_X(self, list_IDs_temp):
        
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.image_path + self.labels[ID]
            data = np.load(file_path)
            X[i,] = data
        return X

    """Generates data batch of size batch_size
       :param list_IDs_temp: list of label ids to load
       :return: batch of masks """
    def _generate_y(self, list_IDs_temp):
        y = np.empty((self.batch_size, self.n_classes, *self.dim))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.mask_path + self.labels[ID]
            data = np.load(file_path)
            # One hot encode ground truth
            data_encoded = to_categorical(data[0,:,:,:], num_classes=5)
            data_encoded = np.rollaxis(data_encoded, 3,0)
            y[i,] = data_encoded
        return y   
    
    """Center crop image
       :param image: loaded image
       :return: centeral cropped image """
    def center_crop(self, image):
        slices = self.slices
        length = self.length
        height = self.height
      
        slice_from = (image.shape[0] - slices) // 2
        length_from = (image.shape[1] - length) //2
        height_from = (image.shape[2] - height) // 2
        
        slice_to = slice_from + slices
        length_to = length_from + length
        height_to = height_from + height
        
        return image[slice_from:slice_to, length_from:length_to, height_from:height_to]
