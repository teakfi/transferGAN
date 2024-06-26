# HDF5 dataformat

The HDF5 is one of the dataformats for high performance computing, widely used in scientific computing.

Documentation for the fileformat is found here: https://www.hdfgroup.org/
Python module information (with nice documentation) is found here: https://www.h5py.org/

Current code expects following data structure:
The data in HDF5 file is supposed to be in root of the file in datasets named: testdata, traindata, rundata, or valdata. At the moment only training is implemented in the code set, so only traindata is being used.
The datasets are in a following format: array(index,width,height)=32bit float from 0 to 1, meaning grayscale. Each array has before and after -(image)values, stacked either next to each other or above each other, the default is before on the left and after on the right so the width is double of the single image.

Future improvements to be implemented are:
multichannel support (3 channels first)
support for data being in a group or subgroup inside of a file

Performance:
I assume that synchronizing the block size in the data file with batch size in training will have the best performance based on the amount of data reading threads being used. With data compression this is a likely place for performance issues.