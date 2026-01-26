'''
This is the module from which we'll run the Python program.
'''
# Torch dependencies
import torch
from torch import optim
import torch.nn as nn
from torch.nn import Sequential
from torch.optim.lr_scheduler import CosineAnnealingLR
# Hyperparameter tuning dependencies:
import argparse
# Local dependencies:
import data
import classifier
import predict
import image_classifier_functions as icf
# Data extraction and storage dependencies:
import os
import subprocess
import sys
# Using the dedicated, lightweight 'wget' module for retrieving the dataset by
# URL:
subprocess.run(['pip', 'install', 'wget'],
               check = True,
               capture_output = True)
import wget

def run_initial_script(cwd):
    '''
    Helper function for efficient data extraction, retrieval, and storage, to
    be run at the beginning of the 'train.py' script.

    Args:
        cwd (str): String representation of the current working directory.
    '''
    command_2 = ['rm', '-r', f'{cwd}/flowers']
    command_3 = ['mkdir', f'{cwd}/flowers']
    command_4 = ['tar', '-xzvf', f'{cwd}/flower_data.tar.gz',
                '-C', f'{cwd}/flowers']
    command_5 = ['rm', 'flower_data.tar.gz']

    # Run the initial script commands, bubbling up any 'subprocess' exceptions
    # for proper program exit:
    try:
        print(f'Retrieving flower data from the cloud server, and removing ' +
              f'previous local version of the flower data directory...')
        wget.download(
            'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
        )
        if os.path.isdir(os.getcwd() + '/flowers'):
            subprocess.run(command_2, check = True, capture_output = True)
        print(f'Flower data successfully retrieved from the cloud server...')
        print(f'Creating directory at "{cwd}/flowers"...')
        subprocess.run(command_3, check = True, capture_output = True)
        print(f'Directory created at: "{cwd}/flowers"...')
        print('Now extracting and retrieving data from "tar.gz" archive...')
        subprocess.run(command_4, check = True, capture_output = True)
        print(f'Successfully extracted and retrieved flower data, stored in' +
            f' directory "{cwd}/flowers...')
        print(f'Removing the "flower_data.tar.gz" archive file from the ' +
              f'project directory...')
        subprocess.run(command_5, check = True, capture_output = True)
        print(f'"flower_data.tar.gz" archive file removed, data loading ' +
              f'completed!')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while optimizing data ' +
              f'preprocessing: {e}')
        sys.exit(1)

if __name__ == '__main__':
    '''
    Parses the command-line arguments and runs the primary logic of the image
    classifier.

    Args:
        args: The list of command-line arguments, which include customizable
        hyperparameter settings.
    '''

    # Customize and store the appropriate command line arguments for training
    # using argparse, which will also handle invalid CLI arguments:
    parser = argparse.ArgumentParser()
    arch_help_str = ('Required, choose from the following architectures ' +
                     'strings: resnet50", "vgg16", "vgg19".')
    # Add the model architecture argument to the parser:
    parser.add_argument('--arch', type = str,
                        help = arch_help_str, required = True)
    # Specify whether to train or perform hyperparameter tuning, making '--hp'
    # a verbose flag:
    parser.add_argument('--hp', action = 'store_true')
    # Add model hyperparameter arguments to the parser, setting default values
    # so the call to 'main' from CLI stays consistent:
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--hunits', type = int, default = 512)
    parser.add_argument('--lr', type = str, default = 0.001)
    parser.add_argument('--dropout', type = str, default = 0.5)
    # '--anneal' does not need a CLI arg passed:
    parser.add_argument('--anneal', action = 'store_true')
    args = parser.parse_args()

    # Typecast the float parser args from strings to floats:
    if args.lr:
        args.lr = float(args.lr)
    if args.dropout:
        args.dropout = float(args.dropout)

    # Run the initial script for optimization of data extraction, retrieval, and
    # storage:
    run_initial_script(os.getcwd())

    # Initialize the data using the instance-stored 'flower_data', keeping its
    # state in a 'Data' object:
    data = data.Data(args.batch_size)

    # Conditioned upon providing the verbose '--hp' CLI flag, either perform
    # hyperparameter tuning or use the training configuration provided by
    # the user:
    if args.hp:
        # Import the 'config' module:
        import config

        # Instantiate the HPTuningConfigs class, for access to config dictionaries
        # as attributes, while hyperparameter tuning:
        hp_configs = config.HPTuningConfigs(data)

        print('"hp" flag signaled, bypassing provided hyperparameter arguments to ' +
              'perform hyperparameter tuning...\n')

        # With respect to the specified architecture, perform advanced scheduler
        # tuning, passing the appropriate config and config tuning function to
        # the 'tune_with_scheduler' AdvancedScheduler instance method:
        scheduler = config.AdvancedScheduler()
        if args.arch == 'resnet50':
            scheduler.tune_with_scheduler(hp_configs.tune_rn50,
                                          hp_configs.rn50_config)
        elif args.arch == 'vgg16':
            scheduler.tune_with_scheduler(hp_configs.tune_vgg16,
                                          hp_configs.vgg16_config)
        elif args.arch == 'vgg19':
            scheduler.tune_with_scheduler(hp_configs.tune_vgg19,
                                          hp_configs.vgg19_config)

    # Otherwise, train the network using the user-defined hyperparameters passed
    # in via CLI.
    else:
        model = icf.get_pretrained_network(args.arch)

        # Use the 'children' method on the image classifier to unpack the layers
        # defined in the Classifier class, replacing the final fully connected
        # classifier or fc layers in the given pre-trained model with our own
        # classifier layers:
        if args.arch == 'resnet50':
            img_classifier = classifier.Classifier(
                in_features = model.fc.in_features,
                dropout_rate = args.dropout,
                hidden_layers = args.hunits
            )
            model.fc = nn.Sequential(
                *list(img_classifier.children())
            )
        else:
            # For the 'vgg16' and 'vgg19' architectures, because the model
            # is defined as an 'nn.Sequential' object, cast it as such, and
            # retrieve the final set of layers from its classifier:
            model.classifier = torch.nn.Sequential(
                *list(model.classifier.children())[:-1]
            )
            img_classifier = classifier.Classifier(
                # Retrieve the input features to the final set of layers of
                # the VGG19 classifier, by retrieving the 'in_features' of its
                # first layer:
                in_features = model.classifier[0].in_features,
                dropout_rate = args.dropout,
                hidden_layers = args.hunits
            )
            model.classifier = nn.Sequential(
                *list(img_classifier.children())
            )

        # Move the model to the GPU, if available:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model = model.to(device)

        criterion = nn.NLLLoss()

        # Use the 'fc' attribute parameters when using Resnet50 architecture.
        if args.arch == 'resnet50':
            optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)
        # For all other architectures, use the 'classifier' attribute parameters
        else:
            optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)

        # If Cosine Annealing on the learning rate is requested:
        if args.anneal:
            # Instatiate the CosineAnnealingLR scheduler:
            lr_scheduler = CosineAnnealingLR(
                optimizer, args.epochs, eta_min = args.lr - 4e-4
            )
        else:
            lr_scheduler = None

        # Affirm commencement of model training:
        print(f'\nCommencing training of the model using the following ' +
              f'pre-trained architecture and hyperparameter configuration: \n')
        for attr, value in vars(args).items():
            print(f'{attr}: {value}')
        print('\n')

        # Iterate through the training loop:
        for epoch in range(args.epochs):
            loss, accuracy = icf.iterate_training_loop(
                data, device, model, optimizer, criterion, epoch, args.epochs,
                args.arch, False,
                scheduler = lr_scheduler if lr_scheduler is not None else None
            )

        # Test the image classification model on the test data:
        icf.test_network(data, device, model, criterion)

        # For ease of inference, store the 'class_to_idx' mapping:
        model.class_to_idx = data.image_datasets['train_dataset'].class_to_idx

        # Save the model as a checkpoint, storing the returned checkpoint file
        # name:
        checkpoint_file = icf.save_checkpoint(
            vars(args), model, optimizer, loss,
            accuracy = accuracy, arch = args.arch
        )

        # Upon saving the checkpoint, run the helper function 'make_prediction'
        # from the 'predict' module to use the model to make a prediction
        # on a stock image:
        predict.make_prediction(
            checkpoint_file, model, optimizer, device
        )