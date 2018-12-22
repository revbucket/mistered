""" Loads the config.json into python readable format """

import json
import os
from NFP.custom_datasets import cifar_subset
import adversarial_perturbations as ap
import adversarial_attacks as aa


config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           'nfp_config6.json'))
config_dir = os.path.dirname(config_path)
config_dict = json.loads(open(config_path, 'rb').read())

dataset = cifar_subset(size=config_dict['subset_size'], mode=config_dict['mode'])
num_dx = config_dict['num_dx']
num_iterations = config_dict['num_iterations']
weight = config_dict['weight']

threat_model = ap.ThreatModel(ap.DeltaAddition, {'lp_style': config_dict['lp_style'],
                                                 'lp_bound': config_dict['lp_bound']})

attack_method = aa.PGD

index = config_dict['index']

step_size = config_dict['step_size']