"""
This module provides functionality for and defines the final fully-connected
layer(s) of the classifier.
"""
# Torch dependencies:
import torch.nn as nn

class Classifier(nn.Module):
    '''
    Represents the final fully-connected layer(s) of the image classifier.
    '''
    def __init__(self, in_features: int, hidden_layers: int,
                dropout_rate, out_features: int = 102):
        '''
        Initializes the final fully-connected layers of the 'Classifier' object.

        Args:
            in_features (int): Final input feature count of the pre-trained
            network being utilized.
            hidden_layers (int): Hyperparameter for tuning the output size of
            'fc1' and input size of 'fc2'
            dropout_rate: Hyperparameter for tuning the non-zero and zero inputs
            to be passed to 'fc2'.
            out_features: Total classes being classified, defaulted to the known
            102 flower classes.
        '''
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_layers)
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.fc2 = nn.Linear(hidden_layers, out_features)
        self.logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        '''
        Required, as a subclass of 'nn.Module', forward pass function, for the
        image classification model.

        Args:
            x (torch.FloatTensor): The batch of images passed in from the
            training loop.
        '''
        x = self.logsoftmax(
                self.fc2(
                    self.dropout(
                        self.relu(
                            self.fc1(x)))))