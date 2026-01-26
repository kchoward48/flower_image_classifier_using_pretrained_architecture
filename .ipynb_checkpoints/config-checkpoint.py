"""
This module initializes and defines hyperparameter tuning using the Ray
Tune framework (Note: Done so in earnest, as manual trial-and-error was
nauseating and I needed a more systematic approach.)
"""
# Torch dependencies:
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# Ray Tune dependences:
from ray import tune
from ray.tune import Tuner, TuneConfig, with_resources, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
# Local dependencies:
import classifier
import image_classifier_functions as icf
# Additional dependencies
import numpy as np
import os

class AdvancedScheduler:
    '''
    Instantiates instance attributes that define advanced scheduling of the
    tuning completed on 'HPTuningConfigs' objects, while also defining instance
    methods for extraction/outputting of best performing configs.
    '''
    def __init__(self):
        '''
        Initializes the ASHAScheduler for early stopping, if hyperparameter
        tuning.
        '''
        self.scheduler = ASHAScheduler(
            metric = 'loss',
            mode = 'min'
        )

        # Further hyperparameter optimization, with respect to validation loss:
        self.search_alg = OptunaSearch(
            metric = 'loss',
            mode = 'min'
        )

    def tune_with_scheduler(self, train_func, config):
        '''
        Instance method with which to initialize and properly configure the
        hyperparameter tuning process, using the scheduler.

        Args:
            train_func (func): Training function handle with which to tune.
            config (dict): The hyperparameter space with which to tune.
        '''

        # Instantiate a new hyperparameter ray tuner:
        tuner = Tuner(
            with_resources(train_func,
                           resources = {'gpu': 1}),
            param_space = config,
            tune_config = TuneConfig(
                search_alg = self.search_alg,
                scheduler = self.scheduler,
                num_samples = -1
            )
        )

        # Proceed with tuning on the hyperparameter config:
        results = tuner.fit()

        # Output the best tuning trial result:
        best_trial = results.get_best_result('loss', 'min', 'last')
        print(f'Best training config: {best_trial.config}')
        # Display the metrics of the best tuning trial result:
        print(f'Best trial final validation loss and accuracy: ' +
            f'{best_trial.metrics}')

class HPTuningConfigs:
    '''
    Defines and instantiates instance attributes for hyperparameter search spaces,
    with respect to the architecture, as well as the instance methods used to
    handle the training of each space.
    '''
    def __init__(self, data):
        '''
        Defines the instance attributes for the data and the hyperparameter
        spaces.

        Args:
            data (dict): Holds the data loaders for each set of the flower data.
        '''
        self.data = data

        # Define the config search space when using the ResNet50 architecture:
        self.rn50_config = {
            # Searching for the number of epochs:
            'epochs': tune.choice([10]),
            # Searching for the batch size:
            'batch_size': tune.choice([64]),
            # Searching for the hidden units size:
            'hunits': tune.choice([2048]),
            # Searching for the learning rate:
            'lr': tune.choice([1e-2]),
            # Searching for optimal regularizing dropout rate:
            'dropout': tune.choice(list(np.linspace(0.27, 0.3, num = 4))),
            # Searching for binary choice of cosine annealing:
            'anneal': tune.choice([True])
        }

        # Define the config search space when using the VGG16 architecture:
        self.vgg16_config = {
            # Searching for the number of epochs:
            'epochs': tune.choice([10]),
            # Searching for the batch size:
            'batch_size': tune.choice([64]),
            # Searching for the hidden units size:
            'hunits': tune.choice([4096]),
            # Searching for the learning rate:
            'lr': tune.choice([1e-3]),
            # Searching for optimal regularizing dropout rate:
            'dropout': tune.choice(list(np.linspace(0.06, 0.1, num = 4))),
            # Searching for binary choice of cosine annealing:
            'anneal': tune.choice([True])
        }

        # Define the config search space when using the VGG19 architecture:
        self.vgg19_config = {
            # Searching for the number of epochs:
            'epochs': tune.choice([10]),
            # Searching for the batch size:
            'batch_size': tune.choice([64]),
            # Searching for the hidden units size:
            'hunits': tune.choice([4096]),
            # Searching for the learning rate:
            'lr': tune.choice([1e-3]),
            # Searching for optimal regularizing dropout rate:
            'dropout': tune.choice(list(np.linspace(0.06, 0.1, num = 4))),
            # Searching for binary choice of cosine annealing:
            'anneal': tune.choice([True])
        }

    def tune_vgg16(
        self, config,
        checkpoint_dir = os.getcwd() + '/checkpoint_configs'
    ):
        '''
        Wrapper function to be invoked when hyperparameter tuning with respect to
        the 'VGG16' architecture.

        Args:
            config (dict): The hyperparameter search space to use for tuning.
            checkpoint_dir (str): Project directory in which to save Ray Tune
            checkpoints.
        '''
        # Retrieve and intialize the pre-trained architecture:
        model = icf.get_pretrained_network('vgg16')

        # Retrieve the final sequence of the 'VGG19' image classifier:
        model.classifier = torch.nn.Sequential(
            *list(model.classifier.children())[:-1]
        )

        # Use the 'Classifier' class to define our final image classifier
        # sequence:
        img_classifier = classifier.Classifier(
            in_features = model.classifier[0].in_features,
            dropout_rate = config['dropout'],
            hidden_layers = config['hunits']
        )

        # Unpack and store it as an appended 'nn.Sequential', using the image
        # classifier sequence's 'children' method:
        model.classifier = nn.Sequential(
            *list(img_classifier.children())
        )

        # Send the model to the assigned GPU:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Define the loss function and optimizer, only optimizing our classifier
        # model layer(s):
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(),
                               lr = config['lr'])

        # If Cosine Annealing on the learning rate is requested:
        if config['anneal']:
            # Instatiate the CosineAnnealingLR scheduler:
            lr_scheduler = CosineAnnealingLR(
                optimizer, config['epochs'],
                eta_min = config['lr'] - 4e-4
            )
        else:
            lr_scheduler = None

        # Using the current config's number of epochs, iteratively perform the
        # training/tuning:
        for epoch in range(config['epochs']):
            loss, accuracy = icf.iterate_training_loop(
                self.data, device, model, optimizer,
                criterion, epoch,
                config['epochs'], 'vgg16', True,
                scheduler = lr_scheduler if lr_scheduler is not None else None
            )

    def tune_vgg19(
        self, config,
        checkpoint_dir = os.getcwd() + '/checkpoint_configs'
    ):
        '''
        Wrapper function to be invoked when hyperparameter tuning with respect to
        the 'VGG19' architecture.

        Args:
            config (dict): The hyperparameter search space to use for tuning.
            checkpoint_dir (str): Project directory in which to save Ray Tune
            checkpoints.
        '''
        # Retrieve and intialize the pre-trained architecture:
        model = icf.get_pretrained_network('vgg19')

        # Retrieve the final sequence of the 'VGG19' image classifier:
        model.classifier = torch.nn.Sequential(
            *list(model.classifier.children())[:-1]
        )

        # Use the 'Classifier' class to define our final image classifier
        # sequence:
        img_classifier = classifier.Classifier(
            in_features = model.classifier[0].in_features,
            dropout_rate = config['dropout'],
            hidden_layers = config['hunits']
        )

        # Unpack and store it as an appended 'nn.Sequential', using the image
        # classifier sequence's 'children' method:
        model.classifier = nn.Sequential(
            *list(img_classifier.children())
        )

        # Send the model to the assigned GPU:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Define the loss function and optimizer, only optimizing our classifier
        # model layer(s):
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(),
                               lr = config['lr'])

        # If Cosine Annealing on the learning rate is requested:
        if config['anneal']:
            # Instatiate the CosineAnnealingLR scheduler:
            lr_scheduler = CosineAnnealingLR(
                optimizer, config['epochs'],
                eta_min = config['lr'] - 4e-4
            )
        else:
            lr_scheduler = None

        # Using the current config's number of epochs, iteratively perform the
        # training/tuning:
        for epoch in range(config['epochs']):
            loss, accuracy = icf.iterate_training_loop(
                self.data, device, model, optimizer,
                criterion, epoch,
                config['epochs'], 'vgg19', True,
                scheduler = lr_scheduler if lr_scheduler is not None else None
            )

    def tune_rn50(
        self, config,
        checkpoint_dir = os.getcwd() + '/checkpoint_configs'
    ):
        '''
        Wrapper function to be invoked when hyperparameter tuning with respect to
        the 'Resnet50' architecture.

        Args:
            config (dict): The hyperparameter search space to use for tuning.
            checkpoint_dir (str): Project directory in which to save Ray Tune
            checkpoints.
        '''
        # Retrieve and intialize the pre-trained architecture:
        model = icf.get_pretrained_network('resnet50')

        # Use the 'Classifier' class to define our final image classifier
        # sequence:
        img_classifier = classifier.Classifier(
            in_features = model.fc.in_features,
            dropout_rate = config['dropout'],
            hidden_layers = config['hunits']
        )

        # Unpack and store it as an appended 'nn.Sequential', using the image
        # classifier sequence's 'children' method:
        model.fc = nn.Sequential(
            *list(img_classifier.children())
        )

        # Define the loss function and optimizer, only optimizing our classifier
        # model layer(s):
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(),
                               lr = config['lr'])

        # If Cosine Annealing on the learning rate is requested:
        if config['anneal']:
            # Instatiate the CosineAnnealingLR scheduler:
            lr_scheduler = CosineAnnealingLR(
                optimizer, config['epochs'],
                eta_min = config['lr'] - 4e-4
            )
        else:
            lr_scheduler = None

        # Send the model to the assigned GPU:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Using the current config's number of epochs, iteratively perform the
        # training/tuning:
        for epoch in range(config['epochs']):
            loss, accuracy = icf.iterate_training_loop(
                self.data, device, model, optimizer,
                criterion, epoch, config['epochs'], 'resnet50', True,
                scheduler = lr_scheduler if lr_scheduler is not None else None
            )