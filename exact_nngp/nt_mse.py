
from jax.example_libraries import optimizers
import jax
import jax.config
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True) # for numerical stability, can disable if not an issue
from jax import numpy as jnp
from jax import scipy as sp
from nt_data import normalize



def softmax(x):
    """Compute softmax values for each row of predictions in x."""
    e_x = jnp.exp(x - jnp.max(x, axis = (1), keepdims=True))
    return e_x / e_x.sum(axis=(1), keepdims=True)


def make_loss_acc_fn(kernel_fn, USE_SOFTMAX = False):

  def loss_acc_fn(x_query, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds):
    k_qt = kernel_fn(normalize(x_query, tar_means, tar_stds), normalize(x_target, tar_means, tar_stds))
    k_qa = kernel_fn(normalize(x_query, alt_means, alt_stds), normalize(x_alt, alt_means, alt_stds))
    k_tt_reg = k_tt + reg * jnp.eye(k_tt.shape[0])
    k_aa_reg = k_aa + reg * jnp.eye(k_aa.shape[0])
    pred_target = jnp.dot(k_qt, sp.linalg.solve(k_tt_reg, y_target, assume_a='pos'))
    pred_alt = jnp.dot(k_qa, sp.linalg.solve(k_aa_reg, y_alt, assume_a='pos'))
    if USE_SOFTMAX == True:
      pred_target = softmax(pred_target)
      pred_alt = softmax(pred_alt)
    mse_loss = - 0.5*jnp.mean((pred_target - pred_alt) ** 2)
    acc = jnp.mean(jnp.argmax(pred_target, axis=1) == jnp.argmax(pred_alt, axis=1))
    return mse_loss, acc

  return loss_acc_fn

def get_update_functions(init_params, kernel_fn, lr, USE_SOFTMAX = False):
  opt_init, opt_update, get_params = optimizers.adam(lr) 
  opt_state = opt_init(init_params)
  loss_acc_fn = make_loss_acc_fn(kernel_fn, USE_SOFTMAX)
  value_and_grad = jax.value_and_grad(lambda params, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds: loss_acc_fn(params['x'],
                                                                       x_target,
                                                                       y_target,
                                                                       x_alt,
                                                                       y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds), has_aux=True)

  @jax.jit
  def update_fn(step, opt_state, params, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds):
    (loss, acc), dparams = value_and_grad(params, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds)
    return opt_update(step, dparams, opt_state), (loss, acc)

  return opt_state, get_params, update_fn

def get_grad(x_query, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds, kernel_fn, USE_SOFTMAX = False):
  loss_acc_fn = make_loss_acc_fn(kernel_fn, USE_SOFTMAX)
  value_and_grad = jax.value_and_grad(lambda params, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds: loss_acc_fn(params['x'],
                                                                       x_target,
                                                                       y_target,
                                                                       x_alt,
                                                                       y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds), has_aux=True)
  params = {'x': x_query}
  (loss, acc), dparams = value_and_grad(params, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds)
  return dparams
