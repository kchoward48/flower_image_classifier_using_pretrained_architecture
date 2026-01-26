"""
This module initializes the flower data being trained, tested, and validated.
"""
# Torch dependencies
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
# Data extraction and storage dependencies:
import os

class Data:
    '''
    Instantiates instance attributes that define preprocessing of data,
    including transforms composition, ImageFolder access of the flower data,
    and loading of data with parameterized batch sizes.
    '''
    def __init__(self, batch_size):
        '''
        Creates all necessary instance attributes for pre-processing data to
        be used.

        Args:
            batch_size (int): Hyperparameter that defines the mini-batch size
            during training, validation, and testing.
        '''
        self.data_dir = os.getcwd() + '/flowers'
        self.train_dir = self.data_dir + '/train'
        self.valid_dir = self.data_dir + '/valid'
        self.test_dir = self.data_dir + '/test'

        # TODO: Define transforms for the training, validation, and
        # testing sets
        self.data_transforms = {
            'train_transforms': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])]),
            'test_transforms': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0225.])]),
            'val_transforms': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0225.])])
        }

        # Define all 'ImageFolder's with which to create training, testing,
        # and validation sets:
        self.image_datasets = {
            'train_dataset': datasets.ImageFolder(
                root = self.train_dir, transform = self.data_transforms['train_transforms']
            ),
            'test_dataset': datasets.ImageFolder(
                root = self.test_dir, transform = self.data_transforms['test_transforms']
            ),
            'valid_dataset': datasets.ImageFolder(
                root = self.valid_dir, transform = self.data_transforms['val_transforms']
            )
        }

        # Define all 'DataLoader's with which to generatively feed forward on
        # the network:
        self.data_loaders = {
            'train_loader': DataLoader(self.image_datasets['train_dataset'],
                                        batch_size = batch_size,
                                        shuffle = True),
            'test_loader': DataLoader(self.image_datasets['test_dataset'],
                                      batch_size = int(batch_size / 2),
                                      shuffle = False),
            'val_loader': DataLoader(self.image_datasets['valid_dataset'],
                                      batch_size = int(batch_size / 2),
                                      shuffle = False)
        }