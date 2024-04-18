from jax.example_libraries import optimizers
import jax
import jax.config
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True) # for numerical stability, can disable if not an issue
from jax import numpy as jnp
from jax import scipy as sp
from nt_data import normalize, one_hot

def cross_entropy(logprobs, targets):
  target_class = jnp.argmax(targets, axis=1)
  pred_class = jnp.argmax(logprobs, axis=1)
  correct_gap = jnp.mean(jnp.abs(target_class - pred_class))
  return correct_gap

def correct_gap(logprobs, targets):
  target_class = jnp.argmax(targets, axis=1)
  nll = jnp.take_along_axis(logprobs, jnp.expand_dims(target_class, axis=1), axis=1)
  ce = -jnp.mean(nll)
  return ce

def make_loss_acc_fn(kernel_fn, OPT_NORM = False):

 
  def loss_acc_fn(x_query, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds):
    if OPT_NORM == False:
      k_qq_tar = kernel_fn(normalize(x_query, tar_means, tar_stds), normalize(x_query, tar_means, tar_stds))
      k_qq_alt = kernel_fn(normalize(x_query, alt_means, alt_stds), normalize(x_query, alt_means, alt_stds))
      k_qt = kernel_fn(normalize(x_query, tar_means, tar_stds), normalize(x_target, tar_means, tar_stds))
      k_qa = kernel_fn(normalize(x_query, alt_means, alt_stds), normalize(x_alt, alt_means, alt_stds))
    else:
      k_qq_tar = kernel_fn(normalize(x_query, tar_means, tar_stds), normalize(x_query, tar_means, tar_stds))
      k_qq_alt = k_qq_tar
      k_qt = kernel_fn(normalize(x_query, tar_means, tar_stds), normalize(x_target, tar_means, tar_stds))
      k_qa = kernel_fn(normalize(x_query, tar_means, tar_stds), normalize(x_alt, alt_means, alt_stds))
    k_tt_reg = k_tt + reg * jnp.eye(k_tt.shape[0])
    k_aa_reg = k_aa + reg * jnp.eye(k_aa.shape[0])
    pred_target = jnp.dot(k_qt, sp.linalg.solve(k_tt_reg, y_target, assume_a='pos'))
    pred_alt = jnp.dot(k_qa, sp.linalg.solve(k_aa_reg, y_alt, assume_a='pos'))
    var_target = k_qq_tar - jnp.dot(k_qt, sp.linalg.solve(k_tt_reg, k_qt.T, assume_a='pos'))
    var_alt = k_qq_alt - jnp.dot(k_qa, sp.linalg.solve(k_aa_reg, k_qa.T, assume_a='pos'))
    kl_loss = - 0.5 * (
      y_alt.shape[1] * jnp.log(jnp.linalg.det(var_alt) / jnp.linalg.det(var_target)) 
      - x_query.shape[0] * y_alt.shape[1] + y_alt.shape[1] * jnp.trace(sp.linalg.solve(var_alt, var_target, assume_a='pos')) 
      + jnp.dot((pred_alt - pred_target).T.reshape(-1), sp.linalg.solve(jnp.kron(jnp.eye(y_alt.shape[1]), var_alt), (pred_alt - pred_target).T.reshape(-1).T, assume_a='pos'))
      )
    acc = cross_entropy(jax.nn.log_softmax(pred_target), pred_alt)
    return kl_loss, acc

  return loss_acc_fn

def get_update_functions(init_params, kernel_fn, lr, OPT_NORM = False):
  opt_init, opt_update, get_params = optimizers.adam(lr) 
  opt_state = opt_init(init_params)
  loss_acc_fn = make_loss_acc_fn(kernel_fn, OPT_NORM)
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

def get_grad(x_query, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds, kernel_fn, OPT_NORM = False):
  loss_acc_fn = make_loss_acc_fn(kernel_fn, OPT_NORM)
  value_and_grad = jax.value_and_grad(lambda params, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds: loss_acc_fn(params['x'],
                                                                       x_target,
                                                                       y_target,
                                                                       x_alt,
                                                                       y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds), has_aux=True)
  params = {'x': x_query}
  (loss, acc), dparams = value_and_grad(params, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, tar_means, tar_stds, alt_means, alt_stds)
  return dparams