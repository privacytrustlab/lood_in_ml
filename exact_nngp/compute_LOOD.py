from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True) # for numerical stability, can disable if not an issue
from jax import numpy as jnp
from jax import scipy as sp
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

import neural_tangents as nt
import nt_data
from nt_models import get_nngp_kernel_fn
import json

import os
from functools import partial
from logs import double_print

from absl import flags
from absl import app

from plot import plot_dist_AUC, plot_images, plot_subsample_images

# from jax_smi import initialise_tracking
import pandas as pd
import shutil
from tqdm import tqdm

# initialise_tracking()


FLAGS = flags.FLAGS



def get_target_data(load_data_from_backup, differ_idx):

    data_x = np.load(f"{load_data_from_backup}/total_dataset.npy")
    data_y = np.load(f"{load_data_from_backup}/total_labels.npy")
    remain_indices = np.arange(data_x.shape[0])
    differing_points = np.transpose(np.array([data_x[differ_idx - 1]]), (0, 2, 3, 1))
    differing_labels = nt_data.one_hot(np.array([data_y[differ_idx - 1]]), 2)
    remain_indices = np.delete(remain_indices, differ_idx - 1)
    x_target_batch = np.transpose(data_x[remain_indices], (0, 2, 3, 1))
    y_target_batch = nt_data.one_hot(data_y[remain_indices], 2)

    return x_target_batch, y_target_batch, differing_points, differing_labels

def cross_entropy(logprobs, targets):
  target_class = jnp.argmax(targets, axis=1)
  nll = jnp.take_along_axis(logprobs, jnp.expand_dims(target_class, axis=1), axis=1)
  ce = -jnp.mean(nll)
  return ce


def mse_loss_acc_fn(x_query, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, kernel_fn):
    
    k_qq_tar = kernel_fn(x_query, x_query)
    k_qq_alt = k_qq_tar
    k_qt = kernel_fn(x_query, x_target)
    k_qa = kernel_fn(x_query, x_alt)
    k_tt_reg = k_tt + reg * jnp.eye(k_tt.shape[0])
    k_aa_reg = k_aa + reg * jnp.eye(k_aa.shape[0])
    pred_target = jnp.dot(k_qt, sp.linalg.solve(k_tt_reg, y_target, assume_a='pos'))
    pred_alt = jnp.dot(k_qa, sp.linalg.solve(k_aa_reg, y_alt, assume_a='pos'))
    mean_loss = - 0.5*jnp.mean((pred_target - pred_alt) ** 2)
    acc = cross_entropy(jax.nn.log_softmax(pred_target), pred_alt)
    return mean_loss, acc

def kl_loss_acc_fn(x_query, x_target, y_target, x_alt, y_alt, k_tt, k_aa, reg, kernel_fn):
    
    k_qq_tar = kernel_fn(x_query, x_query)
    k_qq_alt = k_qq_tar
    k_qt = kernel_fn(x_query, x_target)
    k_qa = kernel_fn(x_query, x_alt)
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




def main(argv):
    del argv
    """# set tf to cpu version because we only use it for datasets"""
    tf.config.experimental.set_visible_devices([], "GPU")

    @jax.jit
    def get_mean_var(x_query, x_target, y_target, x_alt, y_alt, k_tt, k_aa):
        x_query_tar = x_query
        x_query_alt = x_query
        k_qq_tar, k_qq_alt, k_qt, k_qa = KERNEL_FN(x_query_tar, x_query_tar), KERNEL_FN(x_query_alt, x_query_alt), KERNEL_FN(x_query_tar, x_target), KERNEL_FN(x_query_alt, x_alt)
        k_tt_reg = k_tt + FLAGS.REG_CONST * jnp.eye(k_tt.shape[0])
        k_aa_reg = k_aa + FLAGS.REG_CONST * jnp.eye(k_aa.shape[0])
        pred_target, pred_alt = jnp.dot(k_qt, sp.linalg.solve(k_tt_reg, y_target, assume_a='pos')), jnp.dot(k_qa, sp.linalg.solve(k_aa_reg, y_alt, assume_a='pos'))
        var_target, var_alt = k_qq_tar - jnp.dot(k_qt, sp.linalg.solve(k_tt_reg, k_qt.T, assume_a='pos')), k_qq_alt - jnp.dot(k_qa, sp.linalg.solve(k_aa_reg, k_qa.T, assume_a='pos'))
        return pred_target, pred_alt, var_target, var_alt
    
    
    # sanity check for arguments
    if FLAGS.logdir is None:
      raise ValueError("path for storing the results needs to be specified in logdir")
    if FLAGS.DATA_PATH is None:
      raise ValueError("path for loading the larger leave-one-out dataset has to be specified in DATA_PATH")
    if FLAGS.DIFFER_LIST_PATH is None:
      print("Warning: unspecified path for loading the indices of differing data has to be specified in DIFFER_DATA_PATH")
      DIFFER_IND_LIST = [802, 1083, 9063, 5410, 3113]
      print("Loaded default DIFFER_IND_LIST = [802, 1083, 9063, 5410, 3113]")
    else:
      DIFFER_IND_LIST = list(pd.read_csv(FLAGS.DIFFER_LIST_PATH)['differ_idx'])
    
    
    """# Define Kernel"""
    KERNEL_FN = get_nngp_kernel_fn(FLAGS.ARCHITECTURE, FLAGS.DEPTH, FLAGS.WIDTH, FLAGS.ACT, FLAGS.Wstd, FLAGS.bstd, FLAGS.PARAMETERIZATION, 2)
    
    """## log setup"""
    if os.path.isdir(FLAGS.logdir):
      shutil.rmtree(FLAGS.logdir)
    os.makedirs(FLAGS.logdir)

    # create point lood log file
    if FLAGS.TASK == 'log_point':
      query_output_file = open(f'{FLAGS.logdir}/nngp.csv' ,'w')
      query_file_print = partial(double_print, output_file = query_output_file)
      query_file_print("differ_idx,auc,kl,mse")

    for differ_idx in DIFFER_IND_LIST:
      print(f"\n\n****************\nstart plotting differing data {differ_idx}")
      """# Load data"""
      x_target_batch, y_target_batch, DIFFERING_POINTS, DIFFERING_LABELS = get_target_data(FLAGS.DATA_PATH, differ_idx)


      '''Construct the neighboring dataset as training dataset plus the differing point'''
      x_alt_batch = np.append(x_target_batch, DIFFERING_POINTS, axis = 0)
      y_alt_batch = np.append(y_target_batch, DIFFERING_LABELS, axis = 0) 
          
      

      # compute kernel matrix
      k_tt= KERNEL_FN(x_target_batch, x_target_batch)
      k_difdif = KERNEL_FN(DIFFERING_POINTS, DIFFERING_POINTS)
      k_diftar = KERNEL_FN(DIFFERING_POINTS, x_target_batch)
      k_aa = np.block([[k_tt, k_diftar.T], [k_diftar, k_difdif]])

      if FLAGS.TASK == 'log_point':
        # record baseline auc and loss
        added_kl, _ = kl_loss_acc_fn(np.array(DIFFERING_POINTS), x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, KERNEL_FN)  # compute in batches for expensive kernels
        added_mse, _ = mse_loss_acc_fn(np.array(DIFFERING_POINTS), x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, KERNEL_FN)  # compute in batches for expensive kernels
        mean1, mean2, var1, var2 = get_mean_var(np.array(DIFFERING_POINTS), x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa)
        added_auc = plot_dist_AUC(FLAGS.logdir, f"differing_nngp_{differ_idx}", 5000, mean1, mean2, var1, var2, 0)
        query_file_print(f"{differ_idx}, {added_auc}, {-added_kl}, {-added_mse}")
      elif FLAGS.TASK == 'log_landscape':
        # compute landscape around the differing point and save it
        sample_size, im_size, n_channels, num_classes = DIFFERING_POINTS.shape[0], DIFFERING_POINTS.shape[1],DIFFERING_POINTS.shape[3], DIFFERING_LABELS.shape[1]
        grid = np.linspace(-50, 50, 101)
        random_direction = np.random.uniform(-1, 1, (sample_size, im_size, im_size, n_channels))/255
        kl_grid, mse_grid = [], []
        for i in tqdm(grid):
          intercenter = DIFFERING_POINTS + i * random_direction
          kl, _ = kl_loss_acc_fn(intercenter, x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, KERNEL_FN)
          mse, _ = mse_loss_acc_fn(intercenter, x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, KERNEL_FN)
          kl_grid.append(kl)
          mse_grid.append(mse)
        df = pd.DataFrame.from_dict({'pert': grid, 'kl': kl_grid, 'mse': mse_grid})
        df.to_csv(f"{FLAGS.logdir}/landscape_{differ_idx}.csv")

    

if __name__ == '__main__':
  # architecture params
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  flags.DEFINE_string('ARCHITECTURE', 'FC', '@param: FC, Conv, Myrtle, RBF; choice of neural network architecture yielding the corresponding NTK')
  flags.DEFINE_integer('DEPTH', 2, '@param: int; depth of neural network')
  flags.DEFINE_integer('WIDTH', 1024, '@param: int; width of finite width neural network; only used if parameterization is standard')
  flags.DEFINE_string('ACT', 'relu', '@param: [relu, gelu]; choice of the activation function')
  flags.DEFINE_float('Wstd', np.sqrt(2), '@param: float; W std')
  flags.DEFINE_float('bstd', 0, '@param: float; bias std')
  flags.DEFINE_string('PARAMETERIZATION', 'ntk', '@param: [ntk, standard]; whether to use standard or NTK parameterization, see https://arxiv.org/abs/2001.07301')
  flags.DEFINE_float('REG_CONST', 0.05, "@param: float; the regularization or noise level for kernel matrix")
  flags.DEFINE_string('TASK', 'log_point', '@param: [log_point, log_landscape]; whether to compute and log the landscape or not')
  

  # dataset params
  flags.DEFINE_string('DATA_PATH', None, '@param: where to load the fixed dataset as target dataset')
  flags.DEFINE_string('DIFFER_LIST_PATH', None, '@param: where to load the fixed dataset as target dataset')

  # log params
  flags.DEFINE_string('logdir', None, 'Directory where to save evaluation results.')

  app.run(main)