from .common_options import CommonOptions

class TrainOptions(CommonOptions):
    """This class sets up and reads the training options.

    Before setting up and reading training options it sets up and reads the common options defined in the CommonOptions.
    """

    def initialize(self, parser):
        parser = CommonOptions.initialize(self, parser)

        # insert options here with reasonable grouping

        # testing parameters -- these are to be removed after actual parameters are implemented
        parser.add_argument('--training_use_required', required=True, help='required argument')
        parser.add_argument('--training_use_not_required', type=str, default='bbb', help='help text')


        self.isTrain = True
        return parser
    