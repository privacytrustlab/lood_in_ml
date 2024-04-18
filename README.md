# LOOD in ML

This repository contains code to estimate leave-one-out distinguishability (which generalizes the definitions of influence, memorization and information leakage) in machine learning via Gaussian process modelling. 

These estimates are introduced and analyzed in:


- Leave-one-out Distinguishability in Machine Learning [[Paper]](http://arxiv.org/abs/2309.17310)<br>
**Jiayuan Ye, Anastasia Borovykh, Soufiane Hayou, Reza Shokri** <br>
*In International Conference on Learning Representations (**ICLR**) 2024*<br><br>



# Installation


You can install the environment via [this conda specification file](./environment).

- You may also need to set path for tensorflow as `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/`

- To test whether tf path is set right, run `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- You may also need to set `export XLA_PYTHON_CLIENT_PREALLOCATE=false` for running multiple programs on one GPU.

# Running

## Reconstruction between opt query and differing image

To run the query optimization for one randomly sampled differing training data between leave-one-out datasets:
```
cd exact_nngp
python3 nt_opt_query.py --DATASET_SIZE 10000 --DATASET_TYPE balanced --PRIV_LOSS_TYPE kl --ARCHITECTURE FC --DEPTH 5 --QUERY_SIZE 1 --NUM_DIFFER 1 --LEARNING_RATE 3 --NUM_RUNS 1 --REG_CONST 0.1 --DIFFER_TYPE oneclass --NUM_ITERS 600 --INTER_QUERY False --NORM independent --OPT_NORM False --NUM_CLASSES 2 --FIX_DIFFER False --logdir log_gap
```
To evaluate for more differing data records, simply increase the argument `--NUM_RUNS`. The hyperparameter that affects reconstruction quality the most is the regularization constant, i.e., $\sigma$ in the paper. In general, decreasing regularization constant increases the optimized LOOD. It also appears that increasing reg constant makes the optimization problem easier to solve. 


To further plot the gap and the pair of differing image and optimized query, run `exact_nngp/plot_reconstruction.ipynb`.

## Correlation between LOOD and MIA AUC



1. As the comparison baseline, we first measure the empirical leakage for a randomly chosen differing data via MIA on leave-one-out retrained models. To train NN models on the whole dataset, run

```
cd FC_MIA
python3 main.py # train 100 in models
```

2. To train NN models on the leave-one-out dataset which excludes a randomly chosen differing data, run
```
python3 main_out_models.py # train 100 out models for a randomly chosen differing data
```

3. To compute the MIA AUC and prediction difference for the same differing data, run

```
cd FC_MIA
python3 compute_auc_and_diff.py --in_path ./models_in_relu_NTK --out_path ./models_out_relu_NTK --logdir ../log_nngp_vs_nn/log_nn
```

4. The above empirical leakage measurements via retraining NN models on leave-one-out datasets, and by subsequently evaluating MIA, are computationally expensive (which took 3.5 GPU hours for training 200 models). By contrast, we will show that computing the LOOD is efficient (which took 2 GPU minutes) for the same differing data, by running

```
cd exact_nngp
python3 compute_LOOD.py --DATA_PATH ../FC_MIA/models_in_relu_NTK/experiment_0/logs --DIFFER_LIST_PATH ../log_nngp_vs_nn/log_nn/nn.csv --logdir ../log_nngp_vs_nn/log_nngp
```

To evaluate over a larger number of randomly chosen differing data, repeat the above steps (2.3. and 4.) after increasing the range of $i$ in line 69 of [`FC_MIA/main_out_models.py`](./FC_MIA/main_out_models.py). You can then also run `log_nngp_vs_nn/plot_correlation_LOOD.ipynb` to plot the correlation between the MIA performance (evaluated over retrained NN) and LOOD (computed under NNGP) for multiple differing data points.  (To compute correlation, one needs at least two differing data points. To observe statistically significant correlation, we recommend evaluating over at least 50 randomly chosen differing data points.)

## Compare with other influence estimation methods

First, train one model that contains checkpoints for all iterations (this is for computing TracIn score).
```
cd FC_MIA
python3 main_all_checkpoints.py
```

Then run `FC_MIA/plot_compare_influence.ipynb` to compute influence scores estimated using various prior methods as well as mean distance LOOD, and compare their correlation with the ground truth prediction difference between NNs trained on leave-one-out datasets. (To compute correlation, one needs at least two differing data points. To observe statistically significant correlation, we recommend evaluating over at least 50 randomly chosen differing data points.)



## LOOD landscape for specified differing data

To compute the LOOD landscape for differing data with indices in the csv file `../log_nngp_vs_nn/log_nn/nn.csv`, run

```
cd exact_nngp
python3 compute_LOOD.py --DATA_PATH ../FC_MIA/models_in_relu_NTK/experiment_0/logs --logdir ../log_nngp_vs_nn/log_nngp_landscape --TASK log_landscape  --DIFFER_LIST_PATH ../log_nngp_vs_nn/log_nn/nn.csv
```

Proceed to run `exact_nngp/plot_landscape.ipynb`. All the value files and visualizations for LOOD landscape are in the folder `log_nngp_vs_nn/log_nngp_landscape`. (To modify the list of differing data, pass in csv file with other differ_idx values to replace `../log_nngp_vs_nn/log_nn/nn.csv`). 


## MIA performance landscape for specified differing data

Run `FC_MIA/plot_landscape.ipynb`. See all the generated value files and visualizations for MIA AUC landscape in the folder `log_nngp_vs_nn/log_nn_landscape`.

## LOOD and MIA performance comparison under different activation function

To compute MIA performance on NNs with gelu activation, for the same differing data as specified in the experiment for relu activation, run

```
cd FC_MIA_gelu
python3 main.py
python3 main_out_models.py
python3 compute_auc_and_diff.py --in_path ./models_in_gelu_NTK --out_path ./models_out_gelu_NTK --logdir ../log_nngp_vs_nn/log_nn_gelu
```

to (1) train 100 in models with gelu activation (2) train 100 out models with gelu activation for the differing data in the folder `../log_nngp_vs_nn/log_nn/data` (3) compute per-record MIA performance in AUC scores under gelu activation


To compute the LOOD for differing data with the same indices, run

```
cd exact_nngp
python3 compute_LOOD.py --DATA_PATH ../FC_MIA_gelu/models_in_gelu_NTK/experiment_0/logs --DIFFER_LIST_PATH ../log_nngp_vs_nn/log_nn_gelu/nn.csv --logdir ../log_nngp_vs_nn/log_nngp_gelu --ACT gelu
```

Then proceed to `FC_MIA_gelu/plot_activation.ipynb` to plot the LOOD and MIA performance under all evaluated differing data records under gelu activation, and to compare with the results under relu activation.



# Acknowledgements

The code heavily relies on the [Google Neural Tangents Library](https://github.com/google/neural-tangents) for computing the prediction distribution under NNGP.  The code for influence function estimate is adapted from [torch-influence](https://github.com/alstonlo/torch-influence) with significant help from Martin Strobel in terms of reproducing the code.

# Reference
[1] Jiayuan Ye, Anastasia Borovykh, Soufiane Hayou, and Reza Shokri. **Leave-one-out Distinguishability in Machine Learning.**
In *International Conference on Learning Representations (**ICLR**) 2024*.