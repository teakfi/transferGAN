"""Training script for image translation.

This script is for running training of the networks for various purposes with network design and training behaviour are given by options.
"""

from options.train_options import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()
