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
import nt_mse
import nt_kl

import os
from functools import partial
from logs import double_print

from absl import flags
from absl import app

from plot import plot_dist_AUC, plot_images, eval_reconstruction_error

import time
import pandas as pd
import os


FLAGS = flags.FLAGS

def main(argv):
  del argv
  """# set tf to cpu version because we only use it for datasets"""
  tf.config.experimental.set_visible_devices([], "GPU")

  SAVE_LOCATION = f"{FLAGS.logdir}_{FLAGS.PRIV_LOSS_TYPE}_{FLAGS.DATASET}/PARAM-{FLAGS.PARAMETERIZATION}-Wstd-{round(FLAGS.Wstd, 2)}-bstd-{round(FLAGS.bstd, 2)}-ARC-{FLAGS.ARCHITECTURE}-DEPTH-{FLAGS.DEPTH}-ACT-{FLAGS.ACT}-REG-{FLAGS.REG_CONST}-DATASET_SIZE-{FLAGS.DATASET_SIZE}-DATASET_TYPE-{FLAGS.DATASET_TYPE}-NUM_DIFFER-{FLAGS.NUM_DIFFER}-DIFFER_TYPE-{FLAGS.DIFFER_TYPE}-AUGMENT-{FLAGS.AUGMENT}-AUG_STEP-{FLAGS.AUG_STEP}-QUERY-{FLAGS.QUERY_SIZE}-NORM-{FLAGS.NORM}-OPT_NORM-{FLAGS.OPT_NORM}-LR_{FLAGS.LEARNING_RATE}-LR_CEIL_{FLAGS.LR_CEIL:.3g}"

  if not os.path.isdir(SAVE_LOCATION):
    os.makedirs(SAVE_LOCATION)
  

  def get_mean_var(x_query, x_target, y_target, x_alt, y_alt, k_tt, k_aa):
    x_query_tar = nt_data.normalize(x_query, tar_means, tar_stds)
    x_query_alt = nt_data.normalize(x_query, alt_means, alt_stds)
    k_qq_tar, k_qq_alt, k_qt, k_qa = KERNEL_FN(x_query_tar, x_query_tar), KERNEL_FN(x_query_alt, x_query_alt), KERNEL_FN(x_query_tar, nt_data.normalize(x_target, tar_means, tar_stds)), KERNEL_FN(x_query_alt, nt_data.normalize(x_alt, alt_means, alt_stds))
    k_tt_reg = k_tt + FLAGS.REG_CONST * jnp.eye(k_tt.shape[0])
    k_aa_reg = k_aa + FLAGS.REG_CONST * jnp.eye(k_aa.shape[0])
    pred_target, pred_alt = jnp.dot(k_qt, sp.linalg.solve(k_tt_reg, y_target, assume_a='pos')), jnp.dot(k_qa, sp.linalg.solve(k_aa_reg, y_alt, assume_a='pos'))
    var_target, var_alt = k_qq_tar - jnp.dot(k_qt, sp.linalg.solve(k_tt_reg, k_qt.T, assume_a='pos')), k_qq_alt - jnp.dot(k_qa, sp.linalg.solve(k_aa_reg, k_qa.T, assume_a='pos'))
    return pred_target, pred_alt, var_target, var_alt
    
  def pre_train_init():
    # initialize the query
    if FLAGS.DATASET!='census':
      x_init, _ = nt_data.random_init(FLAGS.QUERY_SIZE, DIFFERING_POINTS.shape[1],DIFFERING_POINTS.shape[3], DIFFERING_LABELS.shape[1])
    else:
      x_init = np.random.rand(FLAGS.QUERY_SIZE, DIFFERING_POINTS.shape[1])
    return x_init


  """Define training algorithm for query optimization, here we use adam"""
  def train(PRIV_LOSS_TYPE, X_INIT, save_location, num_train_steps, log_freq=10, seed=1):
    
    
    def predict():
      if FLAGS.ARCHITECTURE != 'Myrtle' and FLAGS.ARCHITECTURE != 'WideResnet' and X_TEST.shape[0] <= 2000:
        xtest, ytest = X_TEST, Y_TEST
        pred_output_file = open('{}/pred_log.txt'.format(save_location) ,'w')
        pred_file_print = partial(double_print, output_file = pred_output_file)
        k_pt = KERNEL_FN(nt_data.normalize(xtest, tar_means, tar_stds), nt_data.normalize(x_target_batch, tar_means, tar_stds))
        # pred_train, pred_test = jnp.dot(k_tt, sp.linalg.solve(k_tt_reg, y_target_batch, assume_a='pos')), jnp.dot(k_pt, sp.linalg.solve(k_tt_reg, y_target_batch, assume_a='pos'))
        k_tt_reg = k_tt + FLAGS.REG_CONST * jnp.eye(k_tt.shape[0])
        pred_test = jnp.dot(k_pt, sp.linalg.solve(k_tt_reg, y_target_batch, assume_a='pos'))
        # train_acc, test_acc = jnp.mean(jnp.argmax(pred_train, axis=1) == jnp.argmax(y_target_batch, axis=1)), jnp.mean(jnp.argmax(pred_test, axis=1) == jnp.argmax(ytest, axis=1))
        test_acc = jnp.mean(jnp.argmax(pred_test, axis=1) == jnp.argmax(ytest, axis=1))
        # pred_file_print(f"train accuracy: {train_acc}\n size of training set: {x_target_batch.shape[0]}\n test accuracy: {test_acc}\n size of test set: {X_TEST.shape[0]}")
        pred_file_print(f"test accuracy: {test_acc}\n size of test set: {xtest.shape[0]}")
    
    def landscape(name, center):
      mse_loss_acc_fn = nt_mse.make_loss_acc_fn(KERNEL_FN, FLAGS.OPT_NORM)
      kl_loss_acc_fn = nt_kl.make_loss_acc_fn(KERNEL_FN, FLAGS.OPT_NORM)
      # compute and record kl, mse
      kl, _ = kl_loss_acc_fn(center, x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)
      mse, _ = mse_loss_acc_fn(center, x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)
      if FLAGS.ARCHITECTURE != 'Myrtle' and FLAGS.ARCHITECTURE != 'WideResnet' and FLAGS.DATASET_SIZE <= 10000:
        kl_grad = nt_kl.get_grad(center, x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds, KERNEL_FN)
        mse_grad = nt_mse.get_grad(center, x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds, KERNEL_FN)

        # compute gradient norm
        kl_grad_norm = np.linalg.norm(kl_grad['x'])
        mse_grad_norm = np.linalg.norm(mse_grad['x'])
      else:
        kl_grad, mse_grad, kl_grad_norm, mse_grad_norm = {'x':None}, {'x':None}, None, None
        
      # compute kernel distance to target dataset and store the matrix
      kern_dist = KERNEL_FN(center, nt_data.normalize(x_target_batch, tar_means, tar_stds))/jnp.sqrt(jnp.kron(np.array([KERNEL_FN(center, center).diagonal()]).T, np.array([k_tt.diagonal()])))

      '''Evaluate the reconstruction error'''
      rec_error = eval_reconstruction_error(DIFFERING_POINTS, center, save_location, name)

      np.savez(f"{save_location}/landscape_{name}.npz", kl = kl, mse = mse, kl_grad_norm = kl_grad_norm, mse_grad_norm = mse_grad_norm, kl_grad = kl_grad['x'], mse_grad = mse_grad['x'], rec_error = rec_error, kern_dist = kern_dist)


    # see if training directory exists
    if not os.path.isdir(save_location):
      os.makedirs(save_location)

    if FLAGS.INIT_METHOD == 'data':
      if FLAGS.INIT_PATH != None:
        params_init = {'x': np.load(FLAGS.INIT_PATH)['best_params_x']}
      else:
        assert(x_target_batch.shape[0]>=DIFFERING_POINTS.shape[0])
        init_indices = np.random.choice(x_target_batch.shape[0], size = DIFFERING_POINTS.shape[0], replace = False)
        params_init = {'x': x_target_batch[init_indices]}
    else:
      params_init = {'x': X_INIT}

    # compute kernel matrix
    k_tt= BATCHED_KERNEL_FN(nt_data.normalize(x_target_batch, tar_means, tar_stds), nt_data.normalize(x_target_batch, tar_means, tar_stds))
    if FLAGS.NORM == 'independent':
      k_difdif = KERNEL_FN(nt_data.normalize(DIFFERING_POINTS, alt_means, alt_stds), nt_data.normalize(DIFFERING_POINTS, alt_means, alt_stds))
      k_diftar = KERNEL_FN(nt_data.normalize(DIFFERING_POINTS, alt_means, alt_stds), nt_data.normalize(x_target_batch, tar_means, tar_stds))
      k_aa = np.block([[k_tt, k_diftar.T], [k_diftar, k_difdif]])
    elif FLAGS.NORM == 'dependent':
      k_aa = BATCHED_KERNEL_FN(nt_data.normalize(x_alt_batch, alt_means, alt_stds), nt_data.normalize(x_alt_batch, alt_means, alt_stds))
    else:
      raise NotImplementedError(f'Unrecognized normalization type {FLAGS.NORM}')
    
    # log setup
    query_output_file = open('{}/training_log.txt'.format(save_location) ,'w')
    query_file_print = partial(double_print, output_file = query_output_file)


    # initalize the privacy loss function and optimizer
    if PRIV_LOSS_TYPE == 'mse':
      loss_acc_fn = nt_mse.make_loss_acc_fn(KERNEL_FN, FLAGS.OPT_NORM)
      train_loss, train_acc = loss_acc_fn(params_init['x'], x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)
      # set learning rate of optimizer to be dependent on the loss at initialization
      opt_state, get_params, update_fn = nt_mse.get_update_functions(params_init, KERNEL_FN, FLAGS.LEARNING_RATE, FLAGS.OPT_NORM)
      query_file_print(f"learning rate: {FLAGS.LEARNING_RATE}, train loss: {train_loss}, train acc: {train_acc}")
    elif PRIV_LOSS_TYPE == 'kl':
      # compute initial loss
      loss_acc_fn = nt_kl.make_loss_acc_fn(KERNEL_FN, FLAGS.OPT_NORM)
      train_loss, train_acc = loss_acc_fn(params_init['x'], x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)
      # set learning rate of optimizer to be dependent on the loss at initialization
      lr_threshold = FLAGS.LR_CEIL
      if (-train_loss) <lr_threshold:
        if train_loss < 0:
          opt_state, get_params, update_fn = nt_kl.get_update_functions(params_init, KERNEL_FN, FLAGS.LEARNING_RATE * 10 * min(np.log10(lr_threshold)-np.log10(-train_loss), 5), FLAGS.OPT_NORM)
          query_file_print(f"learning rate: {FLAGS.LEARNING_RATE * 10 * min(np.log10(lr_threshold) - np.log10(-train_loss), 5)}, train loss: {train_loss}, train acc: {train_acc}")
        else:
          opt_state, get_params, update_fn = nt_kl.get_update_functions(params_init, KERNEL_FN, FLAGS.LEARNING_RATE * 10 * min(np.log10(lr_threshold)-np.log10(train_loss), 5), FLAGS.OPT_NORM)
          query_file_print(f"learning rate: {FLAGS.LEARNING_RATE * 10 * min(np.log10(lr_threshold) - np.log10(train_loss), 5)}, train loss: {train_loss}, train acc: {train_acc}")
      else:
        opt_state, get_params, update_fn = nt_kl.get_update_functions(params_init, KERNEL_FN, FLAGS.LEARNING_RATE, FLAGS.OPT_NORM)
        query_file_print(f"learning rate: {FLAGS.LEARNING_RATE}, train loss: {train_loss}, train acc: {train_acc}")
    else:
      raise NotImplementedError(f'Unrecognized loss type {PRIV_LOSS_TYPE}')
    params = get_params(opt_state)
    
    patience = 0
    patience_tolerance = 20
    for i in range(0,num_train_steps+1):
      last_train_loss = train_loss
      # update parameters either by sampling from target dataset or by optimizing the privacy loss
      if FLAGS.INTER_QUERY == 'False':
        opt_state, (train_loss, train_acc) = update_fn(i, opt_state, params, x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)
        params = get_params(opt_state)
      elif FLAGS.INTER_QUERY == 'True':
        params['x'] = params_init['x'] * (1 - i/num_train_steps) + DIFFERING_POINTS * i/num_train_steps
        train_loss, train_acc = loss_acc_fn(params['x'], x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)
      elif FLAGS.INTER_QUERY == 'Augment':
        params['x'], _ = nt_data.get_augmented_data(DIFFERING_POINTS, DIFFERING_LABELS, tar_means, tar_stds, 1, save_location)
        train_loss, train_acc = loss_acc_fn(params['x'], x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)

      if (-last_train_loss) + FLAGS.LR_CEIL >= (- train_loss):
        patience = patience + 1
      else:
        patience = 0
      if (i % log_freq == 0) or (FLAGS.INTER_QUERY != 'False'):
        query_file_print(f"----step {i}:\n train privacy loss ({PRIV_LOSS_TYPE}): {-train_loss}\n train agreement rate for queries: {train_acc}")
        mean1, mean2, var1, var2 = get_mean_var(params['x'], x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa)
        assert (np.isfinite(mean1).all)
        assert (np.isfinite(mean2).all)
        assert (np.isfinite(var1).all)
        assert (np.isfinite(var2).all)
        auc = plot_dist_AUC(save_location, i, 5000, mean1, mean2, var1, var2)
        kern_dist = KERNEL_FN(nt_data.normalize(DIFFERING_POINTS, alt_means, alt_stds), nt_data.normalize(params['x'], alt_means, alt_stds))/jnp.sqrt(jnp.kron(np.array([KERNEL_FN(nt_data.normalize(DIFFERING_POINTS, alt_means, alt_stds), nt_data.normalize(DIFFERING_POINTS, alt_means, alt_stds)).diagonal()]).T, np.array([KERNEL_FN(nt_data.normalize(params['x'], alt_means, alt_stds), nt_data.normalize(params['x'], alt_means, alt_stds)).diagonal()])))
        np.savez(f"{save_location}/checkpoint_{i}.npz", train_loss = train_loss, params_x = params['x'], mean1 = mean1, mean2 = mean2, var1 = var1, var2 = var2, auc = auc, kern_dist = kern_dist)
        if i == 0:
          best_train_loss, best_params, best_auc = train_loss, params, auc
        elif train_loss < best_train_loss:
          best_train_loss = train_loss
          best_auc = auc
          best_params = params
        if FLAGS.DATASET == 'cifar10':
          subtitles_query = []
          for query_i in range(params['x'].shape[0]):
            query_data = np.array([params['x'][query_i]])
            query_loss, query_acc = loss_acc_fn(query_data, x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)
            subtitles_query.append(f"Single query\nLOOD: {- query_loss:.3f}")
          plot_images(params['x'], f"{save_location}/query_at_time_{i}.png", f"queried images at time {i}\n total LOOD: {-train_loss:.3f}", subtitles_query)
        else:
          query_file_print(f"queried data: {params['x']}")
      if patience >= patience_tolerance and (-train_loss)>1e-5:
        break
        
    
    # record baseline auc and loss
    added_loss, added_acc = loss_acc_fn(np.array(DIFFERING_POINTS), x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)  # compute in batches for expensive kernels
    query_file_print(f"baseline privacy loss ({PRIV_LOSS_TYPE}) on the differing point: {-added_loss}")
    query_file_print(f"baseline squared loss for predictions on leave-one-out dataset: {added_acc}")
    mean1, mean2, var1, var2 = get_mean_var(np.array(DIFFERING_POINTS), x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa)
    base_auc = plot_dist_AUC(save_location, "differing", 5000, mean1, mean2, var1, var2)
    # plot differing image
    # '''Visualize the differing images'''
    subtitles_differ = []
    for differ_i in range(np.array(DIFFERING_POINTS).shape[0]):
      differ_data = np.array([np.array(DIFFERING_POINTS)[differ_i]])
      differ_loss, differ_acc = loss_acc_fn(differ_data, x_target_batch, y_target_batch, x_alt_batch, y_alt_batch, k_tt, k_aa, FLAGS.REG_CONST, tar_means, tar_stds, alt_means, alt_stds)
      if differ_acc == 0:
        subtitles_differ.append(f"Differing data {differ_i}/{np.array(DIFFERING_POINTS).shape[0]}\nLOOD: {- differ_loss:.3f}\n Correct: True")
      else:
        subtitles_differ.append(f"Differing data {differ_i}/{np.array(DIFFERING_POINTS).shape[0]}\nLOOD: {- differ_loss:.3f}\n Correct: False")
    plot_images(np.array(DIFFERING_POINTS), f"{save_location}/differing.png", f"differing Images \n total LOOD: {-added_loss:.4f}", subtitles_differ)
    np.savez(f"{save_location}/best.npz", best_train_loss = best_train_loss, best_params_x = best_params['x'], best_auc = best_auc, base_auc = base_auc, base_loss = added_loss, base_acc = added_acc, tar_means = tar_means, tar_stds = tar_stds)
    
    # '''Evaluate the test performance of the nngp model learned on target dataset'''
    # predict()
    landscape('differ', DIFFERING_POINTS.astype(float))
    if FLAGS.INTER_QUERY == 'False':
      landscape('best', best_params['x'])
    return params, best_params, params_init


  
  """# Define Kernel"""
  KERNEL_FN = get_nngp_kernel_fn(FLAGS.ARCHITECTURE, FLAGS.DEPTH, FLAGS.WIDTH, FLAGS.ACT, FLAGS.Wstd, FLAGS.bstd, FLAGS.PARAMETERIZATION, FLAGS.NUM_CLASSES)
  if FLAGS.ARCHITECTURE == 'Myrtle' or FLAGS.ARCHITECTURE == 'WideResnet':
    FLAGS.USE_BATCHING = 'True'
  if FLAGS.USE_BATCHING == 'True':
    KERNEL_BATCH_SIZE = 10
    BATCHED_KERNEL_FN = nt.batch(KERNEL_FN, KERNEL_BATCH_SIZE)
  else:
    BATCHED_KERNEL_FN = KERNEL_FN
  
  """## Multiple runs of the query optimization"""
  indices_for_differ_pts = list(np.random.permutation(FLAGS.DATASET_SIZE)[:FLAGS.NUM_RUNS])
  for i in indices_for_differ_pts:

    if FLAGS.FIX_DATASET == 'True' and indices_for_differ_pts[0]!=i:
      if FLAGS.DATA_PATH == None:
        FLAGS.DATA_PATH = f"{SAVE_LOCATION}/RUN_{indices_for_differ_pts[0]}"
    if FLAGS.FIX_DIFFER == 'True' and indices_for_differ_pts[0]!=i:
      if FLAGS.DIFFER_PATH == None:
        FLAGS.DIFFER_PATH = f"{SAVE_LOCATION}/RUN_{indices_for_differ_pts[0]}"

    
    """# Load data"""
    x_target_batch, y_target_batch, DIFFERING_POINTS, DIFFERING_LABELS, tar_means, tar_stds, X_TEST, Y_TEST = nt_data.get_target_data(FLAGS.DATASET, FLAGS.DATASET_SIZE, FLAGS.DATASET_TYPE, FLAGS.NUM_DIFFER, FLAGS.DIFFER_TYPE, f"{SAVE_LOCATION}/{FLAGS.RUN_NAME}RUN_{i}", FLAGS.DATA_PATH, FLAGS.DIFFER_PATH, FLAGS.NUM_CLASSES, start_index=i*FLAGS.NUM_DIFFER)

    # Only randomly initialize at each run
    x_init = pre_train_init()

    '''Construct the augmented data points to include in both datasets'''
    for aug_time in range(FLAGS.AUGMENT//FLAGS.AUG_STEP + 1):
      # go to next loop if dataset and aug are size zero
      if aug_time == 0 and FLAGS.DATASET_SIZE == 0:
        continue

      # """# log file setups"""
      if aug_time == 0:
        save_location = f"{SAVE_LOCATION}/{FLAGS.RUN_NAME}RUN_{i}"
      else:
        save_location = f"{SAVE_LOCATION}/{FLAGS.RUN_NAME}RUN_{i}_AUG_{aug_time}"
      if not os.path.isdir(save_location):
        os.makedirs(save_location)
      elif os.path.exists(f"{save_location}/best.npz"):
        continue

      # construct augment images and add it to the target dataset
      if aug_time > 0:
        x_aug, y_aug = nt_data.get_augmented_data(DIFFERING_POINTS, DIFFERING_LABELS, tar_means, tar_stds, FLAGS.AUG_STEP, save_location)
        x_target_batch, y_target_batch = np.append(x_target_batch, x_aug, axis = 0), np.append(y_target_batch, y_aug, axis = 0) 

      '''Construct the neighboring dataset as training dataset plus the differing point'''
      x_alt_batch = np.append(x_target_batch, DIFFERING_POINTS, axis = 0)
      y_alt_batch = np.append(y_target_batch, DIFFERING_LABELS, axis = 0) 
      if FLAGS.NORM == 'dependent':
        alt_means, alt_stds = nt_data.get_normalization_data(x_alt_batch)
      else:
        alt_means, alt_stds = tar_means, tar_stds
      
      """Run the training algorithm to learn QUERY_SIZE number of images. Here we run 300 training steps to get reasonable performance, please train for more steps for full convergence."""
      if FLAGS.PRIV_LOSS_TYPE == 'both':
        _ , best_params, _ = train('kl', x_init, f'{save_location}', FLAGS.NUM_ITERS)
        _ , best_params, _ = train('mse', x_init, f'{save_location}', FLAGS.NUM_ITERS)
      else:
        _ , best_params, _ = train(FLAGS.PRIV_LOSS_TYPE, x_init, f'{save_location}', FLAGS.NUM_ITERS)
      
      

if __name__ == '__main__':
  np.random.seed(8483203)
  # architecture params
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  flags.DEFINE_string('ARCHITECTURE', 'FC', '@param: FC, Conv, Myrtle, RBF; choice of neural network architecture yielding the corresponding NTK')
  flags.DEFINE_integer('DEPTH', 5, '@param: int; depth of neural network')
  flags.DEFINE_integer('WIDTH', 1024, '@param: int; width of finite width neural network; only used if parameterization is standard')
  flags.DEFINE_string('ACT', 'relu', '@param: [relu, gelu]; choice of the activation function')
  flags.DEFINE_float('Wstd', np.sqrt(2), '@param: float; W std')
  flags.DEFINE_float('bstd', 0, '@param: float; bias std')
  flags.DEFINE_string('PARAMETERIZATION', 'ntk', '@param: [ntk, standard]; whether to use standard or NTK parameterization, see https://arxiv.org/abs/2001.07301')
  flags.DEFINE_float('REG_CONST', 1, "@param: float; the regularization or noise level for kernel matrix")

  # dataset params
  flags.DEFINE_string('DATASET', 'cifar10', '@param [cifar10, cifar100, mnist, svhn_cropped')
  flags.DEFINE_integer('NUM_CLASSES', 2, '@param: int; number of classes for classification')
  flags.DEFINE_integer('DATASET_SIZE', 500, '@param: int; number of training images to use')
  flags.DEFINE_string('DATASET_TYPE', 'balanced', '@param: [balanced, oneclass]; how to sample the images that are used as target dataset')
  flags.DEFINE_string('FIX_DATASET', 'True', '@param: whether to fix the dataset')
  flags.DEFINE_string('FIX_DIFFER', 'False', '@param: whether to fix the differing data')
  flags.DEFINE_string('DATA_PATH', None, '@param: where to load the fixed dataset as target dataset')


  # differing and augmentation params
  flags.DEFINE_integer('NUM_DIFFER', 1, '@param: int; number of of test image to use as differing image')
  flags.DEFINE_string('DIFFER_TYPE', 'oneclass', '@param: [balanced, oneclass, sameclass, noise]; what images to use as differing image')
  flags.DEFINE_string('DIFFER_PATH', None, '@param: where to load the differing points')
  flags.DEFINE_integer('AUGMENT', 0, '@param: int; number of data augmentation of the differing point in neighboring datasets')
  flags.DEFINE_integer('AUG_STEP', 2, '@param: int; number of data augmentation of the differing point in neighboring datasets')
  
  # training params
  flags.DEFINE_float('LEARNING_RATE', 20, '@param : float; learning rate for query optimization')
  # for mse loss and RBF kernel
  # flags.DEFINE_float('LEARNING_RATE', 200, '@param : float; learning rate for query optimization')
  flags.DEFINE_integer('QUERY_SIZE', 1, '@param: int; number of images to learn as queries')
  flags.DEFINE_string('PRIV_LOSS_TYPE', 'kl', '@param: [mse, kl, both]; which privacy loss indistinguishability metric to optimize over during training')
  flags.DEFINE_integer('NUM_ITERS', 300, '@param: int; number of iterations for query optimizaiton')
  flags.DEFINE_string('INTER_QUERY', 'True', '@param: [True, False, Augment]; whether to sample the query from the training dataset for training or not')

  # customized optimization
  flags.DEFINE_string('USE_BATCHING', 'False', '@param: [True, False]; where to batch the computation for kernel matrix')
  flags.DEFINE_string('OPT_NORM', 'False', '@param: [True, False]; whether to optimize the normalized data with respect to target dataset')
  flags.DEFINE_float('LR_CEIL', 1e-7, '@param: float; ceiling value for initialization loss when adaptively tuning learning rate')
  flags.DEFINE_string('INIT_METHOD', 'random', '@param: [random, data]; which initialization method to use during training, random starts with noise, data starts with sampled training data records (note that data method only applies to the case when the number of queries is proportional to num_class)')
  flags.DEFINE_string('INIT_PATH', None, '@param: where the initialization vector is stored')
  flags.DEFINE_string('NORM', 'independent', '@param: [dependent, independent], whether to use data-dependent normalization')

  # log params
  flags.DEFINE_string('logdir', 'log', 'Directory where to save checkpoints data.')
  flags.DEFINE_integer("NUM_RUNS", 5, 'Number of repeated runs for query optimization')
  flags.DEFINE_string('RUN_NAME', '', 'name of the runs')
  

  app.run(main)