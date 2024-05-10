from .common_options import CommonOptions

class TrainOptions(CommonOptions):
    """This class sets up and reads the training options.

    Before setting up and reading training options it sets up and reads the common options defined in the CommonOptions.
    """

    def initialize(self, parser):
        parser = CommonOptions.initialize(self, parser)

        # insert options here with reasonable grouping
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan ]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.') # wgan-gp not incorporated
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--update_freq_D', type=int, default=1, help='how many iterations of data between updating discriminator')
        parser.add_argument('--update_freq_G', type=int, default=1, help='how many iterations of data between updating generator')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.isTrain = True
        return parser
    