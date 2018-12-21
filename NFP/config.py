""" Loads the config.json into python readable format """

import json
import os
import NFP.custom_datasets as custom_datasets
import adversarial_perturbations as ap
import adversarial_attacks as aa


config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           'config.json'))
config_dir = os.path.dirname(config_path)
config_dict = json.loads(open(config_path, 'rb').read())


def path_resolver(path):
    if path.startswith('~/'):
        return os.path.expanduser(path)

    if path.startswith('./'):
        return os.path.join(*[config_dir] + path.split('/')[1:])


dataset = custom_datasets(config_dict['subset_size'], config_dict['mode'])
num_dx = config_dict['num_dx']
num_iterations = config_dict['num_iterations']
weight = config_dict['weight']

threat_model = ap.ThreatModel(ap.DeltaAddition, {'lp_style': config_dict['lp_style'],
                                                 'lp_bound': config_dict['lp_bound']})

attack_method = aa.PGD

