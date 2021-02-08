"""
A pure TensorFlow implementation of a convolutional neural network.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

import tensorflow as tf

from cleverhans import initializers
from cleverhans.model import Model


class ModelMLP(Model):
  def __init__(self, scope, nb_classes, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.nb_filters = 64
    
    # Do a dummy run of fprop to make sure the variables are created from
    # the start
    self.fprop(tf.placeholder(tf.float32, [32, 1, 1, 2]))
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      
      y = tf.layers.dense(x, 8, activation=tf.nn.relu,
                          kernel_initializer=initializers.HeReLuNormalInitializer)
      logits = tf.layers.dense(
          tf.layers.flatten(y), self.nb_classes,
          kernel_initializer=initializers.HeReLuNormalInitializer)
    
      return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}


class ModelMLP_dyn(Model):
  def __init__(self, scope, nb_classes, num_dims, **kwargs):
    del kwargs
    Model.__init__(self, scope, nb_classes, locals())
    self.nb_filters = 64
    self.num_neurons = num_dims * 4 
    self.num_dims = num_dims
    
    # Do a dummy run of fprop to make sure the variables are created from
    # the start
    self.fprop(tf.placeholder(tf.float32, [32, 1, 1, self.num_dims]))
    # Put a reference to the params in self so that the params get pickled
    self.params = self.get_params()

  def fprop(self, x, **kwargs):
    del kwargs
    
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      
      y = tf.layers.dense(x, self.num_neurons, activation=tf.nn.relu,
                          kernel_initializer=initializers.HeReLuNormalInitializer)
      logits = tf.layers.dense(
          tf.layers.flatten(y), self.nb_classes,
          kernel_initializer=initializers.HeReLuNormalInitializer)
    
      return {self.O_LOGITS: logits,
              self.O_PROBS: tf.nn.softmax(logits=logits)}
    
    
    
    
    
    
    
