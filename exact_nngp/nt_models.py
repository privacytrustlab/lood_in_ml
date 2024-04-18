import functools

import jax
import jax.config
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True) # for numerical stability, can disable if not an issue
from jax import numpy as jnp
import numpy as np

from neural_tangents import stax

####################################################################
# Exponential kernel

def RBF_Kernel(x, x_prime, gamma = 1):
  x_norm = jnp.sum(x ** 2, axis = tuple(np.arange(x.ndim)[1:]))
  x_prime_norm = jnp.sum(x_prime **2, axis = tuple(np.arange(x.ndim)[1:]))
  dim = np.prod(x[0].shape)
  return jnp.exp(- gamma/dim * (x_norm[:,None] + x_prime_norm[None,:] - 2 * jnp.dot(x.reshape(x.shape[0], dim), x_prime.reshape(x_prime.shape[0], dim).T)))


####################################################################
# Simple FC, Conv and Myrtle

def FullyConnectedNetwork( 
    depth,
    width,
    W_std = np.sqrt(2), 
    b_std = 0.1,
    num_classes = 10,
    parameterization = 'ntk',
    activation = 'relu'):
  """Returns neural_tangents.stax fully connected network."""
  if activation=='relu':
    activation_fn = stax.Relu()
  elif activation=='gelu':
    activation_fn = stax.Gelu()

  dense = functools.partial(
      stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

  layers = [stax.Flatten()]
  for _ in range(depth):
    layers += [dense(width), activation_fn]
  layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std, 
                        parameterization=parameterization)]

  return stax.serial(*layers)

def FullyConvolutionalNetwork( 
    depth,
    width,
    W_std = np.sqrt(2), 
    b_std = 0.1,
    num_classes = 10,
    parameterization = 'ntk',
    activation = 'relu'):
  """Returns neural_tangents.stax fully convolutional network."""
  if activation=='relu':
    activation_fn = stax.Relu()
  elif activation=='gelu':
    activation_fn = stax.Gelu()
  conv = functools.partial(
      stax.Conv,
      W_std=W_std,
      b_std=b_std,
      padding='SAME',
      parameterization=parameterization)
  layers = []
  for _ in range(depth):
    layers += [conv(width, (2, 2)), activation_fn]
  layers += [stax.Flatten(), stax.Dense(num_classes, W_std=W_std, b_std=b_std,
                                        parameterization=parameterization)]

  return stax.serial(*layers)

def MyrtleNetwork(  
    depth,
    width,
    W_std = np.sqrt(2), 
    b_std = 0.1,
    num_classes = 10,
    parameterization = 'ntk',
    activation = 'relu'):
  """Returns neural_tangents.stax Myrtle network."""
  layer_factor = {5: [1, 1, 1], 7: [1, 2, 2], 10: [2, 3, 3]}
  if depth not in layer_factor.keys():
    raise NotImplementedError(
        'Myrtle network with depth %d is not implemented!' % depth)
  activation_fn = stax.Relu()
  layers = []
  conv = functools.partial(
      stax.Conv,
      W_std=W_std,
      b_std=b_std,
      padding='SAME',
      parameterization=parameterization)
  layers += [conv(width, (3, 3)), activation_fn]

  # generate blocks of convolutions followed by average pooling for each
  # layer of layer_factor except the last
  for block_depth in layer_factor[depth][:-1]:
    for _ in range(block_depth):
      layers += [conv(width, (3, 3)), activation_fn]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))]

  # generate final blocks of convolution followed by global average pooling
  for _ in range(layer_factor[depth][-1]):
    layers += [conv(width, (3, 3)), activation_fn]
  layers += [stax.GlobalAvgPool()]

  layers += [
      stax.Dense(num_classes, W_std, b_std, parameterization=parameterization)
  ]

  return stax.serial(*layers)

####################################################################
# More complicated Wide ResNet
def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
  Main = stax.serial(
      stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
      stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
  Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
      channels, (3, 3), strides, padding='SAME')
  return stax.serial(stax.FanOut(2),
                     stax.parallel(Main, Shortcut),
                     stax.FanInSum())

def WideResnetGroup(n, channels, strides=(1, 1)):
  blocks = []
  blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
  for _ in range(n - 1):
    blocks += [WideResnetBlock(channels, (1, 1))]
  return stax.serial(*blocks)

def WideResnet(block_size, k, num_classes):
  return stax.serial(
      stax.Conv(16, (3, 3), padding='SAME'),
      WideResnetGroup(block_size, int(16 * k)),
      WideResnetGroup(block_size, int(32 * k), (2, 2)),
      WideResnetGroup(block_size, int(64 * k), (2, 2)),
      stax.AvgPool((8, 8)),
      stax.Flatten(),
      stax.Dense(num_classes, 1., 0.))

def get_kernel_fn(architecture, depth, width, act, W_std, b_std, parameterization, num_classes):
  if architecture == 'FC':
    return FullyConnectedNetwork(depth=depth, width=width, activation=act, W_std=W_std, b_std=b_std, parameterization=parameterization, num_classes=num_classes)
  elif architecture == 'Conv':
    return FullyConvolutionalNetwork(depth=depth, width=width, activation=act, W_std=W_std, b_std=b_std, parameterization=parameterization, num_classes=num_classes)
  elif architecture == 'Myrtle':
    return MyrtleNetwork(depth=depth, width=width, activation=act, W_std=W_std, b_std=b_std, parameterization=parameterization, num_classes=num_classes)
  elif architecture == 'WideResnet':
    return WideResnet(block_size=4, k=1, num_classes=num_classes)
  else:
    raise NotImplementedError(f'Unrecognized architecture {architecture}')

def get_nngp_kernel_fn(architecture, depth, width, act, W_std, b_std, parameterization, num_classes=10):
  if architecture == 'RBF':
    KERNEL_FN = jax.jit(RBF_Kernel)
  else:
    _, _, kernel_fn = get_kernel_fn(architecture, depth, width, act, W_std, b_std, parameterization, num_classes)
    KERNEL_FN = jax.jit(functools.partial(kernel_fn, get='nngp'))
  return KERNEL_FN