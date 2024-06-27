# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import, division, print_function, \
                       unicode_literals
                       
import tensorflow as tf

from utils import to_tf_tensor, flatten
from itest import ITest #shared
from gmm_data import GMMInput, GMMOutput #shared
from gmm_objective import gmm_objective #shared
from input_utils import read_gmm_instance



class TensorflowGMM(ITest):
    '''Test class for GMM differentiation by Tensorflow using eager
    execution.'''

    def prepare(self, input):
        '''Prepares calculating. This function must be run before
        any others.'''

        self.alphas = to_tf_tensor(input.alphas)
        self.means = to_tf_tensor(input.means)
        self.icf = to_tf_tensor(input.icf)

        self.x = to_tf_tensor(input.x)
        self.wishart_gamma = to_tf_tensor(input.wishart.gamma)
        self.wishart_m = to_tf_tensor(input.wishart.m)

        self.objective = tf.zeros(1)
        self.gradient = tf.zeros(0)

    def output(self):
        '''Returns calculation result.'''

        return GMMOutput(self.objective.numpy(), self.gradient.numpy())

    def calculate_objective(self, times):
        '''Calculates objective function many times.'''

        for _ in range(times):
            self.objective = gmm_objective(
                self.alphas,
                self.means,
                self.icf,
                self.x,
                self.wishart_gamma,
                self.wishart_m
            )


    def calculate_jacobian(self, times):
        '''Calculates objective function jacobian many times.'''

        for _ in range(times):
            with tf.GradientTape(persistent = True) as t:
                t.watch(self.alphas)
                t.watch(self.means)
                t.watch(self.icf)

                self.objective = gmm_objective(
                    self.alphas,
                    self.means,
                    self.icf,
                    self.x,
                    self.wishart_gamma,
                    self.wishart_m
                )

            J = t.gradient(self.objective, (self.alphas, self.means, self.icf))
            self.gradient = tf.concat([ flatten(d) for d in J ], 0)

def calculate_jacobian(file):
    input = read_gmm_instance(file, False)
    te = TensorflowGMM()
    te.prepare(input)
    te.calculate_jacobian(1)
    return te.gradient

def calculate_objective(file):
    input = read_gmm_instance(file, False)
    te = TensorflowGMM()
    te.prepare(input)
    te.calculate_objective(1)
    return te.objective