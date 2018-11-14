""" Code to build a mnist data loader """


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils.pytorch_utils as utils
import config
import mnist_cnn


###############################################################################
#                           PARSE CONFIGS                                     #
###############################################################################

DEFAULT_DATASETS_DIR = config.DEFAULT_DATASETS_DIR
MNIST_WEIGHT_PATH    = config.MODEL_PATH
DEFAULT_BATCH_SIZE   = config.DEFAULT_BATCH_SIZE
DEFAULT_WORKERS      = config.DEFAULT_WORKERS
MNIST_MEANS          = config.MNIST_MEANS
MNIST_STDS           = config.MNIST_STDS
###############################################################################
#                          END PARSE CONFIGS                                  #
###############################################################################


##############################################################################
#                                                                            #
#                               MODEL LOADER                                 #
#                                                                            #
##############################################################################

def load_pretrained_mnist_cnn(return_normalizer=False, manual_gpu=None):
    """ Helper fxn to initialize/load the pretrained mnist cnn
    """

    # Resolve load path
    weight_path = os.path.join(MNIST_WEIGHT_PATH,
                               'mnist.th')

    # Resolve CPU/GPU stuff
    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    if use_gpu:
        map_location = None
    else:
        map_location = (lambda s, l: s)

    state_dict = torch.load(weight_path)
    classifier_net = mnist_cnn.Net()

    classifier_net.load_state_dict(state_dict)

    if return_normalizer:
        normalizer = utils.DifferentiableNormalize(mean=MNIST_MEANS,
                                                   std=MNIST_STDS)
        return classifier_net, normalizer

    return classifier_net




##############################################################################
#                                                                            #
#                               DATA LOADER                                  #
#                                                                            #
##############################################################################

def load_mnist_data(train_or_val, extra_args=None, dataset_dir=None,
                    normalize=False, batch_size=None, manual_gpu=None,
                    shuffle=True):
    """ Builds a MNIST data loader for either training or evaluation of
        MNIST data. See the 'DEFAULTS' section in the fxn for default args
    ARGS:
        train_or_val: string - one of 'train' or 'val' for whether we should
                               load training or validation datap
        extra_args: dict - if not None is the kwargs to be passed to DataLoader
                           constructor
        dataset_dir: string - if not None is a directory to load the data from
        normalize: boolean - if True, we normalize the data by subtracting out
                             means and dividing by standard devs
        manual_gpu : boolean or None- if None, we use the GPU if we can
                                      else, we use the GPU iff this is True
        shuffle: boolean - if True, we load the data in a shuffled order
    """

    ##################################################################
    #   DEFAULTS                                                     #
    ##################################################################
    # dataset directory
    dataset_dir = dataset_dir or DEFAULT_DATASETS_DIR
    batch_size = batch_size or DEFAULT_BATCH_SIZE

    # Extra arguments for DataLoader constructor
    if manual_gpu is not None:
        use_gpu = manual_gpu
    else:
        use_gpu = utils.use_gpu()

    constructor_kwargs = {'batch_size': batch_size,
                          'shuffle': shuffle,
                          'num_workers': DEFAULT_WORKERS,
                          'pin_memory': use_gpu}
    constructor_kwargs.update(extra_args or {})

    # transform chain
    transform_list = []
    transform_list.append(transforms.ToTensor())

    if normalize:
        normalizer = transforms.Normalize(mean=MNIST_MEANS,
                                          std=MNIST_STDS)
        transform_list.append(normalizer)

    transform_chain = transforms.Compose(transform_list)
    # train_or_val validation
    assert train_or_val in ['train', 'val']

    ##################################################################
    #   Build DataLoader                                             #
    ##################################################################
    return torch.utils.data.DataLoader(
            datasets.MNIST(root=dataset_dir, train=train_or_val=='train',
                           transform=transform_chain, download=True),
            **constructor_kwargs)
