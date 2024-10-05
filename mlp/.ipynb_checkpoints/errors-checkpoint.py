# -*- coding: utf-8 -*-
"""Error functions.

This module defines error functions, with the aim of model training being to
minimise the error function given a set of inputs and target outputs.

The error functions will typically measure some concept of distance between the
model outputs and target outputs, averaged over all data points in the data set
or batch.
"""

import numpy as np


class SumOfSquaredDiffsError(object):
    """Sum of squared differences (squared Euclidean distance) error."""

    def __call__(self, outputs, targets):
        """Calculates error function given a batch of outputs and targets.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Scalar error function value.
        """
        #TODO write your code here
        batch_size = outputs.shape[0]
        output_dim = outputs.shape[1]
    
        loss = 0
        for n in range(batch_size):
            per_loss = 0
            for k in range(output_dim):
                per_loss += 0.5 * (outputs[n][k] - targets[n][k]) ** 2
            loss += per_loss

        loss /= batch_size
    
        return loss

    def grad(self, outputs, targets):
        """Calculates gradient of error function with respect to outputs.

        Args:
            outputs: Array of model outputs of shape (batch_size, output_dim).
            targets: Array of target outputs of shape (batch_size, output_dim).

        Returns:
            Gradient of error function with respect to outputs. This should be
            an array of shape (batch_size, output_dim).
        """
        #TODO write your code here
        batch_size = outputs.shape[0]
    
        return (outputs - targets) / batch_size

    def __repr__(self):
        return 'SumOfSquaredDiffsError'
