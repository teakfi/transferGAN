"""
This file is a edited import from pix2pix/cyclegan https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master
edits include joining stuff from image folder stuff
"""

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import h5py
import numpy as np
 

class H5AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.h5file_path =  opt.dataroot  # get the h5file
        self.dataname = opt.phase + "data" # the data should be in dataset named like this
        self.dataset = None
        with h5py.File(self.h5file_path, 'r') as file:
            self.dataset_len = len(file[self.dataname])  # data should be in datasets traindata, testdata, rundata, or valdata
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """

        if self.dataset is None:
            self.dataset = h5py.File(self.h5file_path, 'r')[self.dataname]
        # read a image given a random integer index
        ABarray = self.dataset[index]
        AB_path = self.h5file_path+' '+self.dataname+' '+str(index)
        ABarray = np.transpose(ABarray)   # for some reason pillow transposes the data-array, this removes the transpose
        if self.opt.convert_to_3channel_UINT8:
            AB = Image.fromarray(ABarray).convert('RGB')
        else:
            AB = Image.fromarray(ABarray)
        # split AB image into A and B
        w, h = AB.size
        
        if self.opt.data_alignment in ['AB','BA']: 
            w2 = int(w / 2)
            left = AB.crop((0, 0, w2, h))
            right = AB.crop((w2, 0, w, h))
            if self.opt.data_alignment == 'AB':
                A = left
                B = right
            else:
                A = right
                B = left
        elif self.opt.data_alignment in ['AoverB','BoverA']:
            h2 = int(h / 2)          # tif combination merged images above each other...
            over = AB.crop((0,0,w,h2))
            under = AB.crop((0,h2,w,h))
            if self.opt.data_alignment == 'AoverB':
                A = over
                B = under
            else:
                B = under
                A = over
        else:
            raise ValueError('selected data_aligment not implemented')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)

      
        A_transform = get_transform(self.opt, transform_params)
        B_transform = get_transform(self.opt, transform_params)

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_len
