from __future__ import print_function

import NeuralFP
import os
import NFP.config as config
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from cifar10 import cifar_loader as cl
from loss_functions import NFLoss
import adversarial_perturbations as ap
import adversarial_attacks as aa
import utils.checkpoints as checkpoints
import utils.pytorch_utils as utils
import time
import logging
from loss_functions import RegularizedLoss
from loss_functions import PartialXentropy
from NFP.model import CW2_Net
from NFP.custom_datasets import cifarSubset


def train(batch_size=48):
    # set random seed for reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)  # ignore this if using CPU

    # Match the normalizer using in the official implementation
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5],
                                               std=[1.0, 1.0, 1.0])

    # use resnet32
    classifier_net = cl.load_pretrained_cifar_resnet(flavor=32,
                                                     return_normalizer=False,
                                                     manual_gpu=None)
    # load cifar data
    cifar_train = cl.load_cifar_data('train', batch_size=batch_size)
    cifar_test = cl.load_cifar_data('train', batch_size=256)

    nfp = NeuralFP(classifier_net=classifier_net, num_dx=5, num_class=10, dataset_name="cifar",
                   log_dir="~/Documents/deep_learning/AE/submit/mister_ed/pretrained_model")

    num_epochs = 30
    verbosity_epoch = 5

    train_loss = nn.CrossEntropyLoss()

    logger = nfp.train(cifar_train, cifar_test, normalizer, num_epochs, train_loss,
                       verbosity_epoch)

    return logger


def get_finger_print():
    """restore fingerprints"""
    fingerprint_dir = "/home/tianweiy/Documents/deep_learning/AE/NeuralFP/NFP_model_weights/cifar/eps_0.006/numdx_30/"

    fixed_dxs = np.load(os.path.join(fingerprint_dir, "fp_inputs_dx.pkl"), encoding='bytes')
    fixed_dys = np.load(os.path.join(fingerprint_dir, "fp_outputs.pkl"), encoding='bytes')

    # preprocessing
    fixed_dxs = utils.np2var(np.concatenate(fixed_dxs, axis=0), cuda=True)
    fixed_dys = utils.np2var(fixed_dys, cuda=True)

    return fixed_dxs, fixed_dys


def set_up_logger():
    logger = logging.getLogger('sanity')
    hdlr = logging.FileHandler('/home/tianweiy/Documents/deep_learning/AE/NeuralFP/log/pgd_2000_16_5_testV3.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    return logger


def build_adv_examples(attack_method, classifier_net, normalizer, threat_model, attack_loss, num_iterations, inputs,
                       labels):
    """Program to build adversarial example with the given images and labels"""
    attack_object = attack_method(classifier_net, normalizer, threat_model, attack_loss)
    perturbation_out = attack_object.attack(inputs, labels, num_iterations=num_iterations, verbose=False)
    adv_examples = perturbation_out.adversarial_tensors()
    return adv_examples


def evaluation(dis_adv, dis_real, tau):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for adv in dis_adv:
        if adv > tau:
            true_positive += 1
        else:
            true_negative += 1

    for real in dis_real:
        if real > tau:
            false_positive += 1
        else:
            false_negative += 1

    return true_positive, false_positive, true_negative, false_negative


def build_attack_loss(classifier_net, normalizer, loss, relative_weight):
    vanilla_loss = PartialXentropy(classifier_net, normalizer)
    losses = {'vanilla': vanilla_loss, 'fingerprint': loss}
    scalars = {'vanilla': 1., 'fingerprint': -1 * relative_weight}
    combine_loss = RegularizedLoss(losses=losses, scalars=scalars)

    return combine_loss


def test(test_dataset, num_dx, num_iterations, threat_model, attack_method, weight):
    """
    :param test_dataset: custom subset of cifar10
    :param num_dx: number of fingerprinting directions
    :param attack_loss: A Regularized Loss Object
    :param num_iterations: Iteration Steps for gradient-based Attacks
    :param threat_model: Attack threatmodel
    :param attack_method: Attack Method. PGD/CW2 Whatever works.
    :param weight: relative importance between detection performance and fingerprint matching loss
    :return:
    """
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)  # ignore this if using CPU

    # Match the normalizer using in the official implementation
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5],
                                               std=[1.0, 1.0, 1.0])

    # get the model
    classifier_net = CW2_Net()

    # load the weight
    path = "/home/tianweiy/Documents/deep_learning/AE/NeuralFP/NFP_model_weights/cifar/eps_0.006/numdx_30/ckpt" \
           "/state_dict-ep_80.pth"
    classifier_net.load_state_dict(torch.load(path))
    classifier_net.cuda()
    classifier_net.eval()

    # Original Repo uses pin memory here
    cifar_test = test_dataset

    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])

    fixed_dxs, fixed_dys = get_finger_print()

    reject_thresholds = \
        [0. + 0.001 * i for i in range(0, 2000)]

    loss = NFLoss(classifier_net, num_dx=num_dx, num_class=10, fp_dx=fixed_dxs, fp_target=fixed_dys,
                  normalizer=normalizer)

    logger = set_up_logger()

    logger.info("Use Weight " + str(weight))

    dis_adv = []
    dis_real = []

    for idx, test_data in enumerate(cifar_test, 0):
        inputs, labels = test_data

        # comment this if using CPU
        inputs = inputs.cuda()
        labels = labels.cuda()

        # build up attack loss
        attack_loss = build_attack_loss(classifier_net, normalizer, loss, weight)

        # build adversarial example
        adv_examples = build_adv_examples(attack_method, classifier_net, normalizer, threat_model, attack_loss,
                                          num_iterations, inputs, labels)
        assert adv_examples.size(0) is 1

        # compute adversarial loss
        l_adv = loss.forward(adv_examples, labels)
        loss.zero_grad()

        # compute real image loss
        l_real = loss.forward(inputs, labels)
        loss.zero_grad()

        dis_adv.append(l_adv)
        dis_real.append(l_real)

    # Collect Informations for ROC AUC
    total = len(dis_adv)
    tp = []
    fp = []
    tn = []
    fn = []

    for tau in reject_thresholds:
        true_positive, false_positive, true_negative, false_negative = evaluation(dis_adv, dis_real, tau)

        tp.append(true_positive)
        fp.append(false_positive)
        tn.append(true_negative)
        fn.append(false_negative)

        logger.info("The threshold is " + str(tau))
        logger.info("True Positive is " + str(true_positive / total * 100))
        logger.info("False Positive is " + str(false_positive / total * 100))


if __name__ == '__main__':
    test_dataset = config.dataset
    num_dx = config.num_dx
    num_iterations = config.num_iterations
    threat_model = config.threat_model
    weight = config.weight
    attack_method = config.attack_method

    test(test_dataset, num_dx, num_iterations, threat_model, attack_method, weight)

