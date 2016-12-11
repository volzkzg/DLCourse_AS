import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False, num_of_conv, num_of_affine):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    C, H, W = input_dim

    count = 0
    lastC = C

    for i in xrange(3):
      for j in xrange(3):
        # conv_relu * 3
        count = count + 1
        self.params['W' + str(count)] = np.random.normal(0, weight_scale, (num_filters, lastC, filter_size, filter_size))
        self.params['b' + str(count)] = np.zeros(num_filters)
        self.params['gamma' + str(count)] = np.ones(num_filters)
        self.params['beta' + str(count)] = np.zeros(num_filters)
        lastC = num_filters

    count = count + 1
    self.params['W' + str(count)] = np.random.normal(0, weight_scale, (num_filters * H * W / 64, hidden_dim))
    self.params['b' + str(count)] = np.zeros(hidden_dim)

    count = count + 1
    self.params['W' + str(count)] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b' + str(count)] = np.zeros(num_classes)

    self.params['gamma2'] = np.ones(hidden_dim)
    self.params['beta2'] = np.zeros(hidden_dim)

    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(3)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    mode = 'test' if y is None else 'train'
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores, cache1 = conv_batchnorm_relu_pool_forward(X, W1, b1, conv_param, pool_param, self.params['gamma1'], self.params['beta1'], self.bn_params[0])
    scores, cache2 = affine_batchnorm_relu_forward(scores, W2, b2, self.params['gamma2'], self.params['beta2'], self.bn_params[1])
    scores, cache3 = affine_forward(scores, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}

    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #################o###########################################################

    loss, dsoftmax = softmax_loss(scores, y)

    loss += 0.5 * self.reg * np.sum(W1 * W1)
    loss += 0.5 * self.reg * np.sum(W2 * W2)
    loss += 0.5 * self.reg * np.sum(W3 * W3)

    dlayer3, grads['W3'], grads['b3'] = affine_backward(dsoftmax, cache3)
    dlayer2, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = affine_batchnorm_relu_backward(dlayer3, cache2)
    dlayer1, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_batchnorm_relu_pool_backward(dlayer2, cache1)

    grads['W1'] += self.reg * self.params['W1']
    grads['W2'] += self.reg * self.params['W2']
    grads['W3'] += self.reg * self.params['W3']

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


def conv_batchnorm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  a, batchnorm_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  a, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(a, pool_param)
  cache = (conv_cache, batchnorm_cache, relu_cache, pool_cache)
  return out, cache

def conv_batchnorm_relu_pool_backward(dout, cache):
  conv_cache, batchnorm_cache, relu_cache, pool_cache = cache
  da = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(da, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(da, batchnorm_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  a, fc_cache = affine_forward(x, w, b)
  a, batchnorm_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, batchnorm_cache, relu_cache)
  return out, cache

def affine_batchnorm_relu_backward(dout, cache):
  fc_cache, batchnorm_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = batchnorm_backward_alt(da, batchnorm_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta
