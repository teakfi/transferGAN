"""
This file is a edited import from pix2pix/cyclegan https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master
edits include joining stuff from image folder stuff
"""

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

class AlignedDataset(BaseDataset):
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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
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
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        if self.opt.convert_to_3channel_UINT8:
            AB = Image.open(AB_path).convert('RGB')
        else:
            AB = Image.open(AB_path)
        # split AB image into A and B
        w, h = AB.size
        
        if self.opt.data_aligment in ['AB','BA']: 
            w2 = int(w / 2)
            left = AB.crop((0, 0, w2, h))
            right = AB.crop((w2, 0, w, h))
            if self.opt.data_aligment == 'AB':
                A = left
                B = right
            else:
                A = right
                B = left
        elif self.opt.data_aligment in ['AoverB','BoverA']:
            h2 = int(h / 2)          # tif combination merged images above each other...
            over = AB.crop((0,0,w,h2))
            under = AB.crop((0,h2,w,h))
            if self.opt.data_aligment == 'AoverB':
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
        return len(self.AB_paths)
