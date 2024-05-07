import argparse

class CommonOptions():
    """This class defines the common options for both training and running the TransferGAN networks.


    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized."""
        self.initialized = False

    def initialize(self, parser):
        """Define common options for both training and running."""
        self.initialized = True

        # testing parameters -- these are to be removed after actual parameters are implemented
        parser.add_argument('--common_use_required', required=True, help='required argument')
        parser.add_argument('--common_use_not_required', type=str, default='aaa', help='help text')


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