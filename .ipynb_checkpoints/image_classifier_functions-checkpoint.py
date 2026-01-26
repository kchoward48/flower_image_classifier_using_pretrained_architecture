"""
This module provides critical functionality for model retrieval, training, and
checkpoint saving and loading.
"""
# Torch dependencies:
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import models
# Additional dependencies:
import os
import shelve
from pathlib import Path

def get_pretrained_network(architecture):
    '''
    Prepares the pre-trained network for training, given the training
    configuration.

    Args:
        architecture (str): String representation of the chosen architecture.
    '''

    # Load in the appropriate pre-trained network, using the custom methods
    # from 'torchvision.models':
    if architecture == 'resnet50':
        pre_trained_model = models.resnet50(
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        )
    elif architecture == 'vgg16':
        pre_trained_model = models.vgg16_bn(
            # Utilizing the more modern batch-normalizing VGG16:
            weights = models.VGG16_BN_Weights.DEFAULT
        )
    elif architecture == 'vgg19':
        # Utilizing the more modern batch-normalizing VGG19:
        pre_trained_model = models.vgg19_bn(
            weights = models.VGG19_BN_Weights.DEFAULT
        )

    for param in pre_trained_model.parameters():
        param.requires_grad = False

    return pre_trained_model

def iterate_training_loop(data, device, model, optimizer,
                          criterion, epoch,
                          args_epoch, args_arch, hp_enabled, scheduler = None):
    '''
    Performs iteration through the training loop, respective to the number of
    epochs provided in CLI args, returning the validation loss and accuracy
    after completion of each training epoch.:

    Args:
        data (dict): Stores dataloaders for training, testing, and validation
        sets.
        device (torch.device): CUDA device the model has been passed passed to.
        model (nn.Module): The image classification model.
        optimizer (torch.optim): Optimizer to be applied to the classifier
        layers only.
        criterion (nn.NLLLoss): Loss function for calculating gradients/gradient
        descent, as well as for tracking model performance/metrics.
        epoch (int): Holds the current training epoch.
        args_epoch (int): Epoch passed in from CLI or the hyperparameter config.
        args_arch (str): Architecture passed in from CLI or with respect to the
        hyperparameter config.
        hp_enabled (bool): Sets whether hyperparameter tuning is being performed
        or not.
        scheduler (CosineAnnealingWarmupRestarts): Learning rate scheduler, in
        case cosine annealing is requested.

    Returns:
        val_loss (float): Accumualed validation loss, with respect to each
        training epoch.
        val_accuracy (float): Accumulated validation accuracy, with respect to
        each training epoch.
    '''
    if epoch > 0:
        print(f'\nTraining for epoch {epoch} completed; continuing with ' +
              f'training for epoch {epoch + 1}\n')

    # Initialize the accumulating variables:
    running_loss = 0.0
    val_print = 20
    running_accuracy = 0.0

    # Load in images and labels to be trained on, sending each item from
    # the generator to the GPU, and enumerating the training data loader
    # so Ray tuner can iterate properly, with respect to each batch:
    for batch_idx, batch in enumerate(data.data_loaders['train_loader']):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Do a forward pass through the classifier network:
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        # If cosine annealing is requested:
        if scheduler:
            scheduler.step()

        # Calculate the training metrics, for the current batch:
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim = 1)
        equality = top_class == labels.view(*top_class.shape)

        # Reset the optimizer gradients, from the previous batch:
        optimizer.zero_grad()

        # Use the item from the criterion 'loss' to accumulate each training
        # loss for a given set of batches:
        running_loss += loss.item()
        running_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

        # Run the model on the validation set, having trained 20 more batches in
        # the epoch:
        if batch_idx % val_print == 0:

            # Set the model to 'eval' mode:
            model.eval()

            # Initialze the validation loss and validation accuracy:
            val_loss = 0
            val_accuracy = 0


            # Set the model to 'eval' mode:
            model.eval()

            with torch.no_grad():
                # Load in images and labels from the validation set, sending each
                # item from the generator to the GPU:
                for images, labels in data.data_loaders['val_loader']:
                    images, labels = images.to(device), labels.to(device)

                    # Do a forward pass through the classifier network, storing the
                    # log probabilities:
                    logps = model(images)
                    loss = criterion(logps, labels)

                    # Use the item from the backpropagation 'loss' to accumulate the
                    # running validation loss:
                    val_loss += loss.item()

                    # Use 'torch.exp' to convert the log probabilities back:
                    ps = torch.exp(logps)
                    # Use the 'topk' method on the exponential tensor 'ps', to
                    # calculate validation accuracy:
                    top_ps, top_class = ps.topk(1, dim = 1)
                    # Compare the top classes of each image classified with the
                    # actual image labels:
                    equality = top_class == labels.view(*top_class.shape)
                    # Accumulate the accuracy of the classifier on the validation
                    # set, for each batch in the validation set:
                    val_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

            # Calculate and print the training and validation metrics, upon
            # completion of model validation:
            tot_val_loss = val_loss / len(data.data_loaders["val_loader"])
            final_val_accuracy = (val_accuracy /
                              len(data.data_loaders["val_loader"]) * 100)

            print(f'Epoch {epoch + 1} of {args_epoch}...')
            # Divide the accumulated training loss by the rate at which
            # validation is being performed:
            print(f'Running training loss: {running_loss / val_print:.3f}')
            print(f'Running training accuracy: ' +
                  f'{running_accuracy / val_print * 100:.2f}%')
            print(f'Validation loss: {tot_val_loss:.3f}')
            print(f"Validation accuracy: {final_val_accuracy:.2f}%")

            # If hyperparameter tuning:
            if hp_enabled:
                # Import the 'tune' module from the 'ray' package:
                from ray import tune
                # Report the validation loss and accuracy to the tuner:
                tune.report({'loss': tot_val_loss,
                             'accuracy': final_val_accuracy})

            # Reset running training loss and accuracy, upon completion of
            # validation:
            running_loss = 0
            running_accuracy = 0
            # Setting the model back to 'train' mode, in preparation for another
            # set of 20 batches:
            model.train()

    return tot_val_loss, final_val_accuracy

def dir_empty_excluding_hidden(dir_path):
    """
    Checks if a directory is empty, excluding any hidden files and directories,
    where names start with a dot ('.').

    Args:
        dir_path (str): The file name of the directory to be checked.
    """
    p = Path(dir_path)

    # Check for any non-hidden items, using a boolean filter:
    has_vis_items = any(not item.name.startswith('.') for item in p.iterdir())
    return not has_vis_items

def checkpoint_counter(func):
    '''
    Decorator function to count model checkpoints, with respect to pre-trained
    architecture.
    '''
    def wrapper(*args, **kwargs):
        with shelve.open('wrapper_counts') as shelf:
            if kwargs['arch'] == 'vgg16':
                wrapper.n_vgg16 += 1
                kwargs['_vgg16_count'] = wrapper.n_vgg16
                shelf['_vgg16_count'] = wrapper.n_vgg16
            elif kwargs['arch'] == 'vgg19':
                wrapper.n_vgg19 += 1
                kwargs['_vgg19_count'] = wrapper.n_vgg19
                shelf['_vgg19_count'] = wrapper.n_vgg19
            elif kwargs['arch'] == 'resnet50':
                wrapper.n_rn50 += 1
                kwargs['_rn50_count'] = wrapper.n_rn50
                shelf['_rn50_count'] = wrapper.n_rn50
        return func(*args, **kwargs)
    # Before passing the wrapper back to the decorated function,
    # generatively initialize the wrapper variables, for a given
    # runtime, using the 'wrapper_counts.db' file generated by the
    # file generator from the 'shelve' module:
    with shelve.open('wrapper_counts', flag = 'c') as shelf:
        if dir_empty_excluding_hidden(os.getcwd() + f'/checkpoint_configs'):
            wrapper.n_vgg16 = 0
            wrapper.n_vgg19 = 0
            wrapper.n_rn50 = 0
            shelf['_vgg16_count'] = 0
            shelf['_vgg19_count'] = 0
            shelf['_rn50_count'] = 0
        else:
            wrapper.n_vgg16 = shelf['_vgg16_count']
            wrapper.n_vgg19 = shelf['_vgg19_count']
            wrapper.n_rn50 = shelf['_rn50_count']
    return wrapper

@checkpoint_counter
def save_checkpoint(hyperparameters, model, optimizer, val_loss,
                    accuracy = 0, arch = '',
                    _vgg16_count = 0, _vgg19_count = 0,
                    _rn50_count = 0):
    '''
    Helper function for saving model checkpoints, upon completion of training,
    validation, and testing of the model, with user-defined architecture and
    hyperparameter configuration:

    Args:
        hyperparameters (dict): Hyperparameters specified by the user as CLI
        arguments.
        model (nn.Module): The image classification model.
        optimizer (torch.optim): Optimizer being used on the classifier layers
        of the model.
        val_loss (float): Final validation loss, upon training completion.
        accuracy (float): Final validation accuracy, upon training completion.
        arch (str): Pre-trained architecture being used.
        _vgg16_count (int): Count of optimal VGG16 models.
        _vgg19_count (int): Count of optimal VGG19 models.
        _rn50_count (int): Count of optimal ResNet50 models.

    Returns:
        A 'str' containing the custom checkpoint file name, with respect to the
        pre-trained architecture
    '''

    # Create the 'checkpoint' dictionary:
    checkpoint = {
        'epoch': hyperparameters['epochs'] + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_loss': val_loss,
        'accuracy': accuracy,
        'hyperparameters': hyperparameters
    }

    # Give the model checkpoint file a well-defined name, based on the
    # architecture and the count with respect to the architecture; then
    # return the checkpoint file name, to load for future inference:
    if arch == 'resnet50':
        torch.save(checkpoint, os.getcwd() +
                   f'checkpoint_configs/model_{arch}_{_rn50_count}.pth')
        print(f'Model checkpoint saved at: {os.getcwd()}' +
              f'/checkpoint_configs/model_{arch}_{_rn50_count}.pth')
        return os.getcwd() + f'/checkpoint_configs/model_{arch}_{_rn50_count}.pth'
    elif arch == 'vgg16':
        torch.save(checkpoint, os.getcwd() +
                   f'/checkpoint_configs/model_{arch}_{_vgg16_count}.pth')
        print(f'Model checkpoint saved at: {os.getcwd()}' +
              f'/checkpoint_configs/model_{arch}_{_vgg16_count}.pth')
        return os.getcwd() + f'/checkpoint_configs/model_{arch}_{_vgg16_count}.pth'
    elif arch == 'vgg19':
        torch.save(checkpoint, os.getcwd() +
                   f'/checkpoint_config/model_{arch}_{_vgg19_count}.pth')
        print(f'Model checkpoint saved at: {os.getcwd()}' +
              f'/checkpoint_configs/model_{arch}_{_vgg19_count}.pth')
        return os.getcwd() + f'/checkpoint_configsmodel_{arch}_{_vgg19_count}.pth'

def load_checkpoint(checkpoint_file, model):
    '''
    Helper function for loading model checkpoint, upon saving a model and its
    user-defined architecture and hyperparameter configuration.

    Args:
        checkpoint_file (str): Filename of the model checkpoint.
        model (nn.Module): The image classification model.

    Returns:
        The 'nn.Sequential' image classification model loaded from the
        checkpoint.
    '''

    # Load the saved checkpoint from file:
    checkpoint = torch.load(checkpoint_file, weights_only = True)

    model.load_state_dict(checkpoint['model_state'])

    # Affirm the model has been properly loaded from the checkpoint by printing
    # its hyperparameter configuration, final validation loss, and final
    # validation accuracy:
    print('Loading in model checkpoint for inference, which used the following' +
          ' pre-trained architecture and hyperparameter configuration...\n')
    print(checkpoint['hyperparameters'])
    print(f"\nUpon completion of training, this model's final validation " +
          f"loss was {checkpoint['val_loss']:.3f} with a final validation " +
          f"accuracy of {checkpoint['accuracy']:.2f}%\n")

    # Set the model to 'eval' mode, for inference:
    model.eval()

    return model

def test_network(data, device, model, criterion):
    '''
    Tests the trained network for generalization.

    Args:
        data (dict): Stores dataloaders for training, testing, and validation
        sets.
        device (torch.device): CUDA device the model has been passed to.
        model (nn.Module): The image classification model.
        criterion (nn.NLLLoss): Loss function for calculating gradients/gradient
        descent, as well as for tracking model performance/metrics.
    '''
    # Ensure the model is in evaluation mode:
    model.eval()

    # Initialize the test loss and accuracy:
    test_loss = 0
    accuracy = 0

    # Load in images and labels from the test set, sending each
    # item from the generator to the GPU:
    for images, labels in data.data_loaders['test_loader']:
        images, labels = images.to(device), labels.to(device)

        # Do a forward pass through the classifier network, storing the
        # log probabilities:
        logps = model(images)
        loss = criterion(logps, labels)

        # Use the item from the backpropagation 'loss' to accumulate the
        # running test loss:
        test_loss += loss.item()

        # Use 'torch.exp' to convert the log probabilities back:
        ps = torch.exp(logps)

        # Use the 'topk' method on the exponential tensor 'ps', to
        # calculate class probabilities for the test dataset:
        top_ps, top_class = ps.topk(1, dim = 1)
        # Compare the top classes of each image classified with the
        # actual image labels:
        equality = top_class == labels.view(*top_class.shape)
        # Accumulate the accuracy of the classifier on the validation
        # set, for each batch in the test set:
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    # Print the resulting test metrics:
    print(f'\nTest loss: {test_loss / len(data.data_loaders["test_loader"]):.2f}')
    print(f'Test accuracy: {accuracy / len(data.data_loaders["test_loader"]):.2%}\n')