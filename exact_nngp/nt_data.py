import dataclasses
from typing import Callable, Optional

from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True) # for numerical stability, can disable if not an issue
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from plot import plot_subsample_images, plot_images
from functools import partial
import pandas as pd
from sklearn import preprocessing
from logs import double_print

import os

def get_tfds_dataset(name):
  ds_train, ds_test = tfds.as_numpy(
      tfds.load(
          name,
          split=['train', 'test'],
          data_dir = '../data',
          batch_size=-1,
          as_dataset_kwargs={'shuffle_files': False}))

  return ds_train['image'], ds_train['label'], ds_test['image'], ds_test['label']

def one_hot(x,
            num_classes,
            center=True,
            dtype=np.float32):
  assert len(x.shape) == 1
  one_hot_vectors = np.array(x[:, None] == np.arange(num_classes), dtype)
  if center:
    one_hot_vectors = one_hot_vectors - 1. / num_classes
  return one_hot_vectors

def get_normalization_data(arr):
  channel_means = np.mean(arr, axis=(0, 1, 2))
  channel_stds = np.std(arr, axis=(0, 1, 2))
  return channel_means, channel_stds

def normalize(array, mean, std):
  return (array - mean) / std

def unnormalize(array, mean, std):
  return (array * std) + mean


def class_balanced_sample(sample_size: int, 
                          labels: np.ndarray,
                          *arrays: np.ndarray, **kwargs: int):
  """Get random sample_size unique items consistently from equal length arrays.

  The items are class_balanced with respect to labels.

  Args:
    sample_size: Number of elements to get from each array from arrays. Must be
      divisible by the number of unique classes
    labels: 1D array enumerating class label of items
    *arrays: arrays to sample from; all have same length as labels
    **kwargs: pass in a seed to set random seed

  Returns:
    A tuple of indices sampled and the corresponding sliced labels and arrays
  """
  if labels.ndim != 1:
    raise ValueError(f'Labels should be one-dimensional, got shape {labels.shape}')
  n = len(labels)
  if not all([n == len(arr) for arr in arrays[1:]]):
    raise ValueError(f'All arrays to be subsampled should have the same length. Got lengths {[len(arr) for arr in arrays]}')
  classes = np.unique(labels)
  n_classes = len(classes)
  n_per_class, remainder = divmod(sample_size, n_classes)
  if remainder != 0:
    raise ValueError(
        f'Number of classes {n_classes} in labels must divide sample size {sample_size}.'
    )
  if kwargs.get('seed') is not None:
    np.random.seed(kwargs['seed'])
  inds = np.concatenate([
      np.random.choice(np.where(labels == c)[0], n_per_class, replace=False)
      for c in classes
  ])
  return (inds, labels[inds].copy()) + tuple(
      [arr[inds].copy() for arr in arrays])



def class_unbalanced_sample(sample_sizes: np.ndarray, 
                          labels: np.ndarray,
                          *arrays: np.ndarray, **kwargs: int):
  """Get random sample_size unique items consistently from equal length arrays.

  The items are class_imbalanced with respect to labels. 

  Args:
    sample_sizes: Number of elements to get from each array from arrays. Must have size equal to the number of unique classes
    labels: 1D array enumerating class label of items
    *arrays: arrays to sample from; all have same length as labels
    **kwargs: pass in a seed to set random seed

  Returns:
    A tuple of indices sampled and the corresponding sliced labels and arrays
  """
  if labels.ndim != 1:
    raise ValueError(f'Labels should be one-dimensional, got shape {labels.shape}')
  n = len(labels)
  if not all([n == len(arr) for arr in arrays[1:]]):
    raise ValueError(f'All arrays to be subsampled should have the same length. Got lengths {[len(arr) for arr in arrays]}')
  classes = np.unique(labels)
  n_classes = len(classes)
  if not n_classes == len(sample_sizes):
    raise ValueError(f'Need to specify samples to sample from {n_classes} classes. Got lengths {len(sample_sizes)} for {n_classes} classes')
  n_per_class = [np.sum(labels == class_idx) for class_idx in classes]
  if not all([n_per_class[i] >= sample_sizes[i] for i in range(n_classes)]):
    raise ValueError(
        f'Number of samples to sample from each class must be smaller or equal than the sample size.'
    )
  if kwargs.get('seed') is not None:
    np.random.seed(kwargs['seed'])
  inds = np.concatenate([
      np.random.choice(np.where(labels == classes[i])[0], int(sample_sizes[i]), replace=False)
      for i in range(n_classes)
  ])
  return (inds, labels[inds].copy()) + tuple(
      [arr[inds].copy() for arr in arrays])

def random_init(sample_size: int, im_size: int, n_channels: int, num_classes: int):
  """get sample size and data domain dimension
  """
  # uniform initialization
  return np.random.uniform(0, 255, (sample_size, im_size, im_size, n_channels)), one_hot(np.random.randint(num_classes, size=sample_size), num_classes)

@dataclasses.dataclass
class Augmentor:
  """Class for creating augmentation function."""

  # function applied after augmentations (maps uint8 image to float image)
  # if standard preprocessing, this should be function which does channel-wise
  # standardization
  preprocessing_function: Callable[[np.ndarray], np.ndarray]

  # need this to unnormalize images if they are already normalized
  # before applying augmentations
  channel_means: Optional[np.ndarray] = None
  channel_stds: Optional[np.ndarray] = None

  # Specify these to augment at custom rate
  rotation_range: float = 90.0
  width_shift_range: float = 5.0
  height_shift_range: float = 5.0
  horizontal_flip: bool = True
  channel_shift_range: float = 5.0

  def __post_init__(self):
    self.aug_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=self.rotation_range,
        width_shift_range=self.width_shift_range,
        height_shift_range=self.height_shift_range,
        horizontal_flip=self.horizontal_flip,
        channel_shift_range=self.channel_shift_range,
        preprocessing_function=self.preprocessing_function,
    )

  def __call__(self,
               x: np.ndarray,
               normalized: bool = True,
               seed: Optional[int] = None):
    """Augments a numpy array of images.

    Args:
      x: image array (B,H,W,C)
      normalized: if True, then image is assumed to be standard preprocessed and
        therefore must be unnormalized before augmented
      seed: random seed for augmentations

    Returns:
      augmented images
    """

    if normalized:
      x_raw = unnormalize(x, self.channel_means, self.channel_stds)
    else:
      x_raw = x

    iterator = self.aug_generator.flow(  # pytype: disable=attribute-error
        x_raw,
        batch_size=x_raw.shape[0],
        shuffle=False,
        seed=seed)
    x_aug = next(iterator)
    return x_aug


def get_cifar10_target_data(dataset_size, dataset_type, num_differ, differ_type, save_location, load_data_from_backup, load_differ_from_backup, num_classes, start_index = None):
  X_TRAIN, LABELS_TRAIN, X_TEST, LABELS_TEST = get_tfds_dataset('cifar10')
  if num_classes < 10:
    indices = np.where(LABELS_TRAIN < num_classes)
    X_TRAIN, LABELS_TRAIN = X_TRAIN[indices], LABELS_TRAIN[indices]
    indices = np.where(LABELS_TEST < num_classes)
    X_TEST, LABELS_TEST = X_TEST[indices], LABELS_TEST[indices]
  Y_TRAIN, Y_TEST = one_hot(LABELS_TRAIN, num_classes), one_hot(LABELS_TEST, num_classes) 


  if not os.path.isdir(save_location):
    os.makedirs(save_location)

  if load_data_from_backup != None and load_differ_from_backup!=None:
    data = np.load(f"{load_data_from_backup}/datasets.npz")
    channel_means = data['channel_means']
    channel_stds = data['channel_stds']
    remain_indices = data['remain_indices']
    x_target_batch = X_TRAIN[remain_indices]
    y_target_batch = Y_TRAIN[remain_indices]
    differ_data = np.load(f"{load_differ_from_backup}/datasets.npz")
    differing_points = differ_data['added_x']
    differing_labels = differ_data['added_y']
  elif load_data_from_backup != None and load_differ_from_backup==None:
    data = np.load(f"{load_data_from_backup}/datasets.npz")
    channel_means = data['channel_means']
    channel_stds = data['channel_stds']
    remain_indices = data['remain_indices']
    if start_index != None:
      differ_indices = remain_indices[start_index:start_index+num_differ]
    else:
      differ_indices = np.random.choice(range(len(remain_indices)), num_differ)
    target_indices = np.delete(remain_indices, differ_indices)
    x_target_batch = X_TRAIN[target_indices]
    y_target_batch = Y_TRAIN[target_indices]
    differing_points = X_TRAIN[differ_indices]
    differing_labels = Y_TRAIN[differ_indices]
  elif load_data_from_backup == None and load_differ_from_backup!=None:
    differ_data = np.load(f"{load_differ_from_backup}/datasets.npz")
    differing_points = differ_data['added_x']
    differing_labels = differ_data['added_y']
    # """Subsample a batch of smaller training dataset as the (smaller) target dataset"""
    if dataset_type == 'balanced':
      remain_indices, label_for_indices, x_target_batch, y_target_batch = class_balanced_sample(dataset_size, LABELS_TRAIN, X_TRAIN, Y_TRAIN)
    elif differ_type == 'sameclass' and dataset_type == 'oneclass':
      vec_size = np.zeros(num_classes)
      idx_data_class = np.argmax(differing_labels[0])
      vec_size[idx_data_class] = dataset_size
      remain_indices, label_for_indices, x_target_batch, y_target_batch = class_unbalanced_sample(vec_size, LABELS_TRAIN, X_TRAIN, Y_TRAIN)
    else:
      raise NotImplementedError(f'Unrecognized dataset type {dataset_type}')
    channel_means, channel_stds = get_normalization_data(x_target_batch)
  else:
    # """Subsample a batch of smaller training dataset as the (smaller) target dataset"""
    if dataset_type == 'balanced':
      remain_indices, label_for_indices, x_target_batch, y_target_batch = class_balanced_sample(dataset_size, LABELS_TRAIN, X_TRAIN, Y_TRAIN)
    elif dataset_type == 'oneclass':
      vec_size = np.zeros(num_classes)
      idx_data_class = np.random.randint(0, 10)
      vec_size[idx_data_class] = dataset_size
      remain_indices, label_for_indices, x_target_batch, y_target_batch = class_unbalanced_sample(vec_size, LABELS_TRAIN, X_TRAIN, Y_TRAIN)
    else:
      raise NotImplementedError(f'Unrecognized dataset type {dataset_type}')
    channel_means, channel_stds = get_normalization_data(x_target_batch)
    
    # """Construct the differing data points to include in the (larger) alternative dataset"""
    if differ_type == 'balanced':
      _, _, differing_points, differing_labels = class_balanced_sample(num_differ, LABELS_TEST, X_TEST, Y_TEST)
    elif differ_type == 'oneclass':
      vec_size = np.zeros(num_classes)
      idx_differ_class = np.random.randint(0, num_classes)
      vec_size[idx_differ_class] = num_differ
      _, _, differing_points, differing_labels = class_unbalanced_sample(vec_size, LABELS_TRAIN, X_TRAIN, Y_TRAIN)
    elif differ_type == 'sameclass' and dataset_type == 'oneclass':
      idx_differ_class = idx_data_class
      vec_size[idx_differ_class] = num_differ
      _, _, differing_points, differing_labels = class_unbalanced_sample(vec_size, LABELS_TRAIN, X_TRAIN, Y_TRAIN)
    elif differ_type == 'noise':
      differing_points, differing_labels = random_init(num_differ, X_TRAIN.shape[1],X_TRAIN.shape[3], num_classes)
    else:
      raise NotImplementedError(f'Unrecognized differing images type {differ_type}')

  # '''Store the remain_indices of records in the dataset'''
  np.savez(f"{save_location}/datasets.npz", remain_indices = remain_indices, added_x = np.array(differing_points), added_y = np.array(differing_labels), channel_means = channel_means, channel_stds = channel_stds)

  return x_target_batch, y_target_batch, differing_points, differing_labels, channel_means, channel_stds, X_TEST, Y_TEST



'''For cencus dataset'''

def get_census_target_data(dataset_size, dataset_type, num_differ, differ_type, save_location, load_data_from_backup, load_differ_from_backup):
  census_data = pd.read_csv('../data/Census/adultdata.txt', header = None)
  census_data.columns = ['age', 'workclass', 'fnlwgt','education','education-num','marital-status','occupation',\
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

  census_data = census_data.replace({' ?': np.nan})
  census_data.dropna()

  # Drop other columns if we want 
  to_drop = ['sex','education', 'marital-status','occupation',\
            'relationship', 'race', 'native-country', 'fnlwgt', 'capital-gain', 'capital-loss','workclass']
  census_data = census_data.drop(to_drop, axis = 1)

  # Do a one-hot encoding of the categoric variables
  to_encode = []
  for cat in to_encode:
      census_data = pd.concat([census_data, pd.get_dummies(census_data[cat])], axis=1)
  # Drop old columns
  census_data = census_data.drop(to_encode, axis = 1)

  # Do a transformation to numeric values
  to_numeric = []
  for num in to_numeric:
      # get unique values for that category
      unique_vals_num = census_data[num].unique()
      census_data[num].replace(unique_vals_num,
                          [i for i in range(len(unique_vals_num))], inplace=True)

  # Change the salary to a numerical thing
  census_data.loc[(census_data['salary'] == ' <=50K'),'salary']=0
  census_data.loc[(census_data['salary'] == ' >50K'),'salary']=1
  num_classes = 2

  # Subsample the dataset for GP speedup
  indices = np.arange(dataset_size)
  census_data_train = census_data.iloc[indices]
  census_data_test = census_data.iloc[dataset_size:dataset_size + 1000]

  # This setup simply deletes one point from x_full
  x_target_batch = census_data_train.loc[:, census_data_train.columns != 'salary'].to_numpy()
  y_target_batch = census_data_train['salary'].to_numpy()
  X_TEST = census_data_test.loc[:, census_data_test.columns != 'salary'].to_numpy()
  Y_TEST = census_data_test['salary'].to_numpy()
  Y_TEST = one_hot(Y_TEST, num_classes)
  # Normalisation
  scaler1 = preprocessing.StandardScaler()
  x_target_batch = scaler1.fit_transform(x_target_batch)
  channel_means = scaler1.mean_
  channel_stds = np.sqrt(scaler1.var_)
  X_TEST = scaler1.transform(X_TEST)
  # construct differing point
  index_target = 992
  indices = np.delete(indices, obj=index_target, axis=0)
  differing_points = x_target_batch[index_target:index_target+1,:]
  differing_labels = y_target_batch[index_target:index_target+1]
  differing_labels = one_hot(differing_labels, num_classes)
  x_target_batch = np.delete(x_target_batch, obj=index_target, axis=0)
  y_target_batch = np.delete(y_target_batch, obj=index_target, axis=0)
  y_target_batch = one_hot(y_target_batch, num_classes)

  # '''Visualize the (subsampled) target training dataset'''
  if not os.path.isdir(save_location):
    os.makedirs(save_location)
  # log setup
  query_output_file = open('{}/data_log.txt'.format(save_location) ,'w')
  query_file_print = partial(double_print, output_file = query_output_file)
  query_file_print(f"=======================\n(Subsampled) images in the smaller training dataset:\n{x_target_batch}\n")

  # '''Visualize the differing images'''
  query_file_print(f"=======================\nDiffering Images:\n{differing_points}\n")
  
  # '''Store the indices of records in the dataset'''
  np.savez(f"{save_location}/datasets.npz", remain_indices = indices, added_x = np.array(differing_points), added_y = np.array(differing_labels), channel_means = channel_means, channel_stds = channel_stds)


  return x_target_batch, y_target_batch, differing_points, differing_labels, channel_means, channel_stds, X_TEST, Y_TEST

def get_target_data(name, dataset_size, dataset_type, num_differ, differ_type, save_location, load_data_from_backup, load_differ_from_backup, num_classes = 10, start_index = None):
  # """# Load Training Data"""
  if os.path.exists(f"{save_location}/datasets.npz"):
    load_data_from_backup = save_location
    load_differ_from_backup = save_location

  if name == 'cifar10':
    x_target_batch, y_target_batch, differing_points, differing_labels, channel_means, channel_stds, X_TEST, Y_TEST = get_cifar10_target_data(dataset_size, dataset_type, num_differ, differ_type, save_location, load_data_from_backup, load_differ_from_backup, num_classes, start_index)
  elif name == 'census':
    x_target_batch, y_target_batch, differing_points, differing_labels, channel_means, channel_stds, X_TEST, Y_TEST = get_census_target_data(dataset_size, dataset_type, num_differ, differ_type, save_location, load_data_from_backup, load_differ_from_backup)

  return x_target_batch, y_target_batch, differing_points, differing_labels, channel_means, channel_stds, X_TEST, Y_TEST


def get_augmented_data(differing_points, differing_labels, channel_means, channel_stds, aug_times, save_location):
  if aug_times == 0:
    return None, None
  pre_func = partial(normalize, mean = channel_means, std = channel_stds)
  AUG = Augmentor(pre_func, channel_means=channel_means, channel_stds = channel_stds)
  x_aug = AUG(differing_points, normalized=True)
  y_aug = differing_labels
  for aug_time in range(aug_times - 1):
    x_aug = np.append(x_aug, AUG(differing_points, normalized=True), axis = 0)
    y_aug = np.append(y_aug, differing_labels, axis = 0)
  
  # '''Visualize the augmented images'''
  plot_images(x_aug, f"{save_location}/new_augment.png", f"New Augmented Images")
  # Store the augmented images
  np.savez(f"{save_location}/new_augment.npz", x_aug = x_aug, y_aug = y_aug, channel_means = channel_means, channel_stds = channel_stds)
  return x_aug, y_aug