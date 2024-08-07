from .common_options import CommonOptions

class TrainOptions(CommonOptions):
    """This class sets up and reads the training options.

    Before setting up and reading training options it sets up and reads the common options defined in the CommonOptions.
    """

    def initialize(self, parser):
        parser = CommonOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        # insert options here with reasonable grouping
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['vanilla','lsgan'],help='the type of GAN objective. [vanilla| lsgan ]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.') # wgan-gp not incorporated
        parser.add_argument('--lr_policy', type=str, default='linear',choices=['linear','step','plateau','cosine'], help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--update_freq_D', type=int, default=1, help='how many iterations of data between updating discriminator')
        parser.add_argument('--update_freq_G', type=int, default=1, help='how many iterations of data between updating generator')
        parser.add_argument('--beta1D', type=float, default=0.5, help='momentum term of adam for discirminator (1)')
        parser.add_argument('--lrD', type=float, default=0.0002, help='initial learning rate for adam, discriminator (1)')
        parser.add_argument('--beta1G', type=float, default=0.5, help='momentum term of adam for generator')
        parser.add_argument('--lrG', type=float, default=0.0002, help='initial learning rate for adam, generator')
        parser.add_argument('--beta1D2', type=float, default=0.5, help='momentum term of adam for discriminator 2') 
        parser.add_argument('--lrD2', type=float, default=0.0002, help='initial learning rate for adam, discriminator 2')
        parser.add_argument('--beta1E', type=float, default=0.5, help='momentum term of adam for encoder')
        parser.add_argument('--lrE', type=float, default=0.0002, help='initial learning rate for adam, encoder')
        parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for |B-G(A, E(B))|')  # bicycleGAN
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight on D loss. D(G(A, E(B)))')  # bicycleGAN
        parser.add_argument('--lambda_GAN2', type=float, default=1.0, help='weight on D2 loss, D(G(A, random_z))')  # bicycleGAN
        parser.add_argument('--lambda_z', type=float, default=0.5, help='weight for ||E(G(random_z)) - random_z||')  # bicycleGAN
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')  # bicycleGAN
        parser.add_argument('--use_same_D', action='store_true', help='if two Ds share the weights or not')     # bicycleGAN    
         
        self.isTrain = True
        return parser
    