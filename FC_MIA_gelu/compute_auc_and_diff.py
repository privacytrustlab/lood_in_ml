import matplotlib.pyplot as plt
import numpy as np
import os
from functools import partial
import sys
import random
import pandas as pd
import shutil

from scipy.stats import multivariate_normal

from sklearn.metrics import auc, roc_curve
import json
import torch
from tqdm import tqdm


from absl import flags
from absl import app
FLAGS = flags.FLAGS


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def store_data_and_predictions():
    if os.path.exists(FLAGS.logdir):
        shutil.rmtree(FLAGS.logdir)
    os.mkdir(FLAGS.logdir)
    os.mkdir(f'{FLAGS.logdir}/data')
    os.mkdir(f'{FLAGS.logdir}/in_predictions')
    os.mkdir(f'{FLAGS.logdir}/out_predictions')


    # compute predictions for all differing data
    in_dict = {}
    out_dict = {}
    in_folder_list = sorted(os.listdir(FLAGS.in_path))
    out_folder_list = sorted(os.listdir(FLAGS.out_path))
    differ_data_dict = {}
    if in_folder_list == None or out_folder_list == None:
        raise NotImplementedError(f'empty folder to plot')
    
    # compute out predictions
    print("computing out predictions")
    for folder in tqdm(out_folder_list):
        if os.path.exists(f"{FLAGS.out_path}/{folder}/models/model-best.pkl"):
            with open(f"{FLAGS.out_path}/{folder}/params.json", 'r') as f:
                data = json.load(f)
            # read differing point index
            differ_idx = data['differ_data']
            # log differing data value
            differ_data = np.load(f'{FLAGS.out_path}/{folder}/logs/differ_data.npy')
            # compute the out prediction for the differing data
            temp_model = torch.load(f'{FLAGS.out_path}/{folder}/models/model-best.pkl')
            temp_model.eval()
            preds = temp_model(to_cuda(torch.from_numpy(differ_data).float()))
            if differ_idx not in out_dict:
                # log differing point index
                differ_data_dict[differ_idx] = differ_data
                out_dict[differ_idx] = preds.cpu().detach().numpy()
            else:
                out_dict[differ_idx] = np.concatenate((out_dict[differ_idx], preds.cpu().detach().numpy()), axis = 0)
        else:
            print(f"no model for {FLAGS.out_path}/{folder}/models/model-best.pkl")
            

    # compute in predictions
    print("computing in predictions")
    for i in tqdm(range(len(in_folder_list))):
        folder = in_folder_list[i]
        # check if it is full dataset training
        with open(f"{FLAGS.in_path}/{folder}/params.json", 'r') as f:
            data = json.load(f)
        assert(data['differ_data']==None)
        temp_model = torch.load(f'{FLAGS.in_path}/{folder}/models/model-best.pkl')
        temp_model.eval()
        if i ==0:
            for differ_idx in out_dict:
                differ_data = differ_data_dict[differ_idx]
                preds = temp_model(to_cuda(torch.from_numpy(differ_data).float()))
                in_dict[differ_idx] = preds.cpu().detach().numpy()
        else:
            for differ_idx in out_dict:
                differ_data = differ_data_dict[differ_idx]
                preds = temp_model(to_cuda(torch.from_numpy(differ_data).float()))
                in_dict[differ_idx] = np.concatenate((in_dict[differ_idx], preds.cpu().detach().numpy()), axis = 0)


    # storing predictions for all differing points
    print("storing data and predictions")
    for differ_idx in tqdm(in_dict):
        np.save(f'{FLAGS.logdir}/data/differ_idx_{differ_idx}.npy', differ_data_dict[differ_idx])
        in_preds = in_dict[differ_idx]
        np.save(f'{FLAGS.logdir}/in_predictions/differ_idx_{differ_idx}.npy', in_preds)
        out_preds = out_dict[differ_idx]
        np.save(f'{FLAGS.logdir}/out_predictions/differ_idx_{differ_idx}.npy', out_preds)
    
def compute_auc_diff():
    # storing predictions for all differing points
    print("storing data and predictions")
    differ_indices = [int(diff_str.replace('differ_idx_','').replace('.npy', '')) for diff_str in os.listdir(f'{FLAGS.logdir}/data')]
    in_shapes_list = []
    out_shapes_list = []
    auc_list = []
    auc_std_list = []
    diff_list = []
    diff_std_list = []
    for differ_idx in tqdm(differ_indices):
        in_preds = np.load(f'{FLAGS.logdir}/in_predictions/differ_idx_{differ_idx}.npy')
        out_preds = np.load(f'{FLAGS.logdir}/out_predictions/differ_idx_{differ_idx}.npy')
        in_shapes_list.append(in_preds.shape[0])
        out_shapes_list.append(out_preds.shape[0])
        # compute AUC score
        if out_preds.shape[0]==100:
            assert (in_preds.shape[0]==out_preds.shape[0])
            auc_run_list = []
            diff_run_list = []
            for run_i in range(50):
                eval_indices = np.random.choice(out_preds.shape[0], out_preds.shape[0]//2, replace=False)
                ref_indices = np.array(list(set(range(out_preds.shape[0])) - set(eval_indices)))
                # use half of the preds to estimate mean and covariance matrix
                in_preds_reference = in_preds[ref_indices]
                in_preds_eval = in_preds[eval_indices]
                out_preds_reference = out_preds[ref_indices]
                out_preds_eval = out_preds[eval_indices]
                # compute mean and covariance matrix
                in_mean = np.mean(in_preds_reference, axis = 0)
                in_cov = np.cov(in_preds_reference.T)
                out_mean = np.mean(out_preds_reference, axis = 0)
                out_cov = np.cov(out_preds_reference.T)
                # compute auc score
                fpr, tpr, thresholds = roc_curve(np.concatenate((np.ones(in_preds_eval.shape[0]), np.zeros(out_preds_eval.shape[0])), axis = 0), multivariate_normal.pdf(np.append(in_preds_eval, out_preds_eval, axis = 0), in_mean, in_cov)/ (multivariate_normal.pdf(np.append(in_preds_eval, out_preds_eval, axis = 0), out_mean, out_cov) + 1e-30))
            
                # fpr, tpr, thresholds = roc_curve(np.concatenate((np.ones(in_preds_eval.shape[0]), np.zeros(out_preds_eval.shape[0])), axis = 0), np.append(np.max(in_preds_eval, axis=1), np.max(out_preds_eval, axis=1), axis = 0))
                auc_score = auc(fpr, tpr)
                auc_run_list.append(auc_score)
                diff_run_list.append(np.mean((np.mean(in_preds_eval, axis = 0) - np.mean(out_preds_eval, axis = 0))**2))
            auc_list.append(np.mean(np.array(auc_run_list)))
            auc_std_list.append(np.std(np.array(auc_run_list)))
            diff_list.append(np.mean(diff_run_list))
            diff_std_list.append(np.std(diff_run_list))
        else:
            auc_list.append(np.nan)
            auc_std_list.append(np.nan)
            diff_list.append(np.nan)
            diff_std_list.append(np.nan)

    sorted_differ_indices = np.argsort(differ_indices)
    differ_indices = [differ_indices[i] for i in sorted_differ_indices]
    in_shapes_list = [in_shapes_list[i] for i in sorted_differ_indices]
    out_shapes_list = [out_shapes_list[i] for i in sorted_differ_indices]
    auc_list = [auc_list[i] for i in sorted_differ_indices]
    auc_std_list = [auc_std_list[i] for i in sorted_differ_indices]
    diff_list = [diff_list[i] for i in sorted_differ_indices]
    diff_std_list = [diff_std_list[i] for i in sorted_differ_indices]


    df = pd.DataFrame.from_dict({'differ_idx': differ_indices, 'in_shape': in_shapes_list, 'out_shape': out_shapes_list, 'auc': auc_list, 'auc_std': auc_std_list, 'diff': diff_list, 'diff_std': diff_std_list})
    df.to_csv(f'{FLAGS.logdir}/nn.csv')

    
def main(argv):
    del argv
    # # store data and predictions
    store_data_and_predictions()
    compute_auc_diff()

if __name__ == '__main__':
    flags.DEFINE_string('in_path', None, 'path to read the in models')
    flags.DEFINE_string('out_path', None, 'path to read the out models')
    flags.DEFINE_string('logdir', None, 'path to log the output files')

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    app.run(main)
    




