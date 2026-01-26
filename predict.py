'''
This module contains helper functions for making a prediction using the
classifier network.
'''
# torch dependencies:
import torch
from torchvision import transforms
# Image processing and display dependencies:
from PIL import Image
# Additional dependencies:
import os
import json
import numpy as np
# Local dependencies:
import image_classifier_functions as icf


def process_image(image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model.

    Args:
        image (str): File path for the image to be processed.

    Returns:
        numpy.ndarray, representing the transformed image.
    '''
    # Perform appropriate image transformations.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                              std = [0.229, 0.224, 0.225]),
    ])

    # Process the PIL image for use in the model:
    pil_image = Image.open(image).convert('RGB')

    return transform(pil_image)

def make_prediction(checkpoint_file, model, optimizer, device, topk = 5):
    '''
    Predicts the class (or classes) of an image using the trained network.

    Args:
        checkpoint_file (str): Name of the checkpoint file in which model
        configuration is kept.
        model (torchvision.models): The image classification model.
        optimizer: (torch.optim): The optimizer working on the fully-connected
        layers.
        device (torch.device): The cuda GPU being utilized.
        topk (int): Number of top classes the model predicted.
    '''

    # Load the saved checkpoint, passing the file name, model and optimizer to
    # the helper function, and providing a handle to the model:
    model = icf.load_checkpoint(checkpoint_file, model)

    print('Now performing inference using the model by making a prediction ' +
          'a stock flower image...\n')
    # Use the image-processing functions in this 'predict' module to ensure the
    # image being passed via path is valid for predictions with the network:
    image = process_image(
        os.getcwd() + '/istockphoto-1273007054-612x612.jpg'
    )

    orig_image_shape = image.shape

    # Reshape into the expected 4D tensor, for the model.
    image = image.reshape(1, *image.shape)

    # With the image processed and displayed properly, use the network to make a
    # prediction on the image:

    # Send the image to the GPU, and obtain the log probabilities:
    image = image.to(device)
    logps = model(image)

    # Undo the log probabilities using 'torch.exp':
    ps = torch.exp(logps)
    # Use the 'topk' method on the exponential tensor 'ps' to calculate
    # the probabilities of the top predicted classes:
    top_ps, top_classes = ps.topk(topk, dim = 1)

    # Retrieve and load in the label mapping from the provided JSON file.
    json_path = os.getcwd() + '/cat_to_name.json'
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)

    # Invert the 'class_by_idx' dictionary:
    rev_class_to_idx = {int_idx: cls for cls, int_idx in model.class_to_idx.items()}

    # Iteratively retrieve and add each top class name to a tuple, using the
    # 'cat_to_name' JSON mapping and the inverted 'class_by_idx' dictionary:
    top_class_names = tuple()
    for int_idx in top_classes.view((topk,)).cpu().detach().numpy():
        top_class_names = (
            top_class_names + (cat_to_name[rev_class_to_idx[int_idx]],)
        )

    # Zip the top classes and their probabilities:
    top_classes_with_probs = zip(
        # Detach the 'top_ps' tensor from the GPU, and convert it into a
        # numpy array and then into a tuple:
        tuple(top_ps.view((topk,)).cpu().detach().numpy()),
        top_class_names
    )

    # Use the zip generator to iteratively print out the prediction results:
    for i, (prob, top_class) in enumerate(top_classes_with_probs, 1):
        print(f'The class with the #{i} highest probability of being ' +
              f'contained in the image is: {top_class}, with a probability of' +
              f' {prob:.2%}.')