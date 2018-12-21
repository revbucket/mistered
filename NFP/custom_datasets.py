from __future__ import print_function

import torch

from PIL import Image
import cifar10.cifar_loader as cl
import torch.utils.data as data


class cifarSubset(data.Dataset):
    """Custom cifar subset for different evaluation experiments"""

    def __init__(self, mode = 'val', size=100, shuffle=True):
        """
        :param size: size of the custom dataset
        :param shuffle: set to True if we only traverse the datset once, False otherwise
        """
        self.size = size
        data = iter(cl.load_cifar_data(mode, batch_size=size))
        self.inputs, self.labels = next(data)

    def __getitem__(self, index):
        img = self.inputs[index]
        target = self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image(copy from pytorch source)
        # img = Image.fromarray(img)

        return img, target

    def __len__(self):
        return self.size


# Testing Scripts
def test():
    cifar_test = cifarSubset()

    for idx, data in enumerate(cifar_test, 0):
        img, label = data
        print(img.shape)
        print(label.shape)
        # break


if __name__ == '__main__':
    test()
