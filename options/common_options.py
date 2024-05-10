import argparse

class CommonOptions():
    """This class defines the common options for: training, testing and running the TransferGAN networks.


    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized."""
        self.initialized = False

    def initialize(self, parser):
        """Define common options for: training, testing and running."""
        self.initialized = True

        # run information
        parser.add_argument('--name', type=str, default='experiment', help='Name for the run, used in saving information')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        # system information
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        
        # data parameters -- where data is, how preprocess before image
        parser.add_argument('--convert_to_3channel_UINT8', action='store_true', help='If this is selected data is converted to "regular" 3-channel RGB')
        parser.add_argument('--dataroot', required=True, help='Path to input images')
        parser.add_argument('--load_size', type=int, default=256, help='Preprocess scaling for images') # pix2pix/cyclegan default 286
        parser.add_argument('--crop_size', type=int, default=256, help='Preprocess cropping for images after scaling')
        parser.add_argument('--batch_size', type=int, default=1, help='Batch size for input')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),help='maximum number of samples allowed per dataset. If more data present only subset of it is being used')
        parser.add_argument('--dataset_mode', type=str, default='aligned',help='The dataset mode, "aligned" is first to be implemented') # other pix2pix/cyclegan modes may be implemented later on
        parser.add_argument('--direction', type=str, default='AtoB',help='Direction of the data [AtoB | BtoA]')
        parser.add_argument('--data_alignment', type=str, default='AoverB', help='How data is aligned in joined image [AB | BA | AoverB | BoverA]') # my own data went by accident like this
        parser.add_argument('--preprocess', type=str, default='none', help='Preprocessing [resize_and_crop | crop | scale_width | scale_width_and_crop | none]') # preprocessing is usually bad for "real" data like physics or medical, implementation for these others are from pix2pix/cyclegan
        parser.add_argument('--phase', type=str, default='train', help='Operating phase, train is first to be implemented') # test, validation and run are to be implemented
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')

        # network properties
        parser.add_argument('--input_nc', type=int, default=3, help='Number of input channels') # not limited to 1 or 3, but free
        parser.add_argument('--output_nc', type=int, default=3, help='Number of output channels') # not limited to 1 or 3, but free
        parser.add_argument('--model', type=str, default='transfer', help='chooses which model to use. [transfer | pix2pix ]') # other models from pix2pix/cyclegan may appear later
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
   
       
        return parser

    def gather_options(self):
        """Initializes parser with options"""

        if not self.initialized:   # check if initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print options

        This will print both current options and default options if default is not selected.
        """

        message = ''
        message += '----------- Options -----------\n'

        for key, value in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(key), str(value), comment)
        message += '------------- End -------------'
        print(message)

    def parse(self):
        """Parse options and setup"""
        opt = self.gather_options()
        opt.isTrain = self.isTrain      # train or run

        self.print_options(opt)

        self.opt = opt

        return self.opt