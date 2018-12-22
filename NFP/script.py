from __future__ import print_function

import sys

sys.path.append('/home/tianweiy/Documents/deep_learning/AE/submit/mister_ed')

import os
import NFP.nfp_config as config
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
from NFP.custom_datasets import cifar_subset
import matplotlib.pyplot as plt




def get_finger_print():
    """restore fingerprints"""
    fingerprint_dir = "/home/tianweiy/Documents/deep_learning/AE/NeuralFP/NFP_model_weights/cifar/eps_0.006/numdx_30/"

    fixed_dxs = np.load(os.path.join(fingerprint_dir, "fp_inputs_dx.pkl"), encoding='bytes')
    fixed_dys = np.load(os.path.join(fingerprint_dir, "fp_outputs.pkl"), encoding='bytes')

    # preprocessing
    fixed_dxs = utils.np2var(np.concatenate(fixed_dxs, axis=0), cuda=True)
    fixed_dys = utils.np2var(fixed_dys, cuda=True)

    return fixed_dxs, fixed_dys


def set_up_logger(index):
    logger = logging.getLogger('sanity')
    hdlr = logging.FileHandler('/home/tianweiy/Documents/deep_learning/AE/NeuralFP/log/' + index + ".log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    return logger


def build_adv_examples(attack, classifier_net, normalizer, threat_model, attack_loss, num_iterations, length,
    inputs, labels):
    """Program to build adversarial example with the given images and labels"""
    attack_object = attack(classifier_net, normalizer, threat_model, attack_loss)
    perturbation_out = attack_object.attack(inputs, labels, num_iterations=num_iterations, step_size=length,
                                            verbose=False)
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
    scalars = {'vanilla': 1., 'fingerprint': -1. * relative_weight}
    combine_loss = RegularizedLoss(losses=losses, scalars=scalars)

    return combine_loss


def get_prediction(normalizer, examples, classifier_net):
    """Find the label of current image"""
    if normalizer is not None:
        examples = normalizer.forward(examples)

    logits = classifier_net.forward(examples)
    yhat = F.softmax(logits, dim=1)
    pred = yhat.data.max(1, keepdim=True)[1]

    return pred


def test(dataset, num_dx, num_iterations, threat_model, attack_method, weight, step_size, logger):
    """
    :param step_size: Attack Step Size. Should set to really small value
    :param logger: Universal Logger
    :param dataset: custom subset of cifar10
    :param num_dx: number of fingerprinting directions
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
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])

    fixed_dxs, fixed_dys = get_finger_print()

    reject_thresholds = \
        [0. + 0.001 * i for i in range(0, 2000)]

    loss = NFLoss(classifier_net, num_dx=num_dx, num_class=10, fp_dx=fixed_dxs, fp_target=fixed_dys,
                  normalizer=normalizer)

    logger.info("Use Weight " + str(weight))

    dis_adv = []
    dis_real = []

    success_attack = 0
    success_classify = 0

    for idx, test_data in enumerate(dataset, 0):
        inputs, labels = test_data

        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)

        # comment this if using CPU
        inputs = inputs.cuda()
        labels = labels.cuda()

        # build up attack loss
        attack_loss = build_attack_loss(classifier_net, normalizer, loss, weight)

        # build adversarial example
        adv_examples = build_adv_examples(attack_method, classifier_net, normalizer, threat_model, attack_loss,
                                          num_iterations, step_size, inputs, labels)
        assert adv_examples.size(0) is 1

        # compute adversarial loss
        l_adv = loss.forward(adv_examples, labels)

        loss.zero_grad()

        # compute real image loss
        l_real = loss.forward(inputs, labels)

        loss.zero_grad()

        # get the label of real and adversarial images
        adv_class = get_prediction(normalizer, adv_examples, classifier_net)
        real_class = get_prediction(normalizer, inputs, classifier_net)

        if labels.cpu().numpy() != adv_class.cpu().numpy() and labels.cpu().numpy() == real_class.cpu().numpy():
            print("Success Attack")
            success_attack += 1

        if labels.cpu().numpy() == real_class.cpu().numpy():
            print("Success Classify")
            success_classify += 1

        dis_adv.append(l_adv)
        dis_real.append(l_real)

    print("Attack Success: ", success_attack, "/ Total: ", len(dis_adv))
    print("Classify Success: ", success_classify, "/ Total: ", len(dis_real))

    mean_adv = sum(dis_adv) / len(dis_adv)
    min_adv = min(dis_adv)
    max_adv = max(dis_adv)

    mean_real = sum(dis_real) / len(dis_real)
    min_real = min(dis_real)
    max_real = max(dis_real)

    logger.info("Mean Adv: " + str(mean_adv))
    logger.info("Min Adv: " + str(min_adv))
    logger.info("Max Adv: " + str(max_adv))

    logger.info("Mean Real: " + str(mean_real))
    logger.info("Min Real: " + str(min_real))
    logger.info("Max Real: " + str(max_real))


if __name__ == '__main__':
    test_dataset = config.dataset
    num_x = config.num_dx
    num_iter = config.num_iterations
    threat = config.threat_model
    gamma = config.weight
    attack = config.attack_method
    index = config.index
    step_length = config.step_size

    log = set_up_logger(index)

    test(dataset=test_dataset, num_dx=num_x, num_iterations=num_iter, threat_model=threat,
         attack_method=attack, weight=gamma, step_size=step_length, logger=log)
