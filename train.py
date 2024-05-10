"""Training script for image translation.

This script is for running training of the networks for various purposes with network design and training behaviour are given by options.
"""

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)