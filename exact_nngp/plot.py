import matplotlib.pyplot as plt
import numpy as np
import os
from logs import double_print
from functools import partial
import sys
import random
import pandas as pd
from tqdm import tqdm
import shutil


from scipy.stats import multivariate_normal
from sklearn.metrics import auc, roc_curve

np.set_printoptions(precision=3)


SMALL_SIZE = 12
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

IMAGE_SIZE = 3

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


# plot the images
def plot_images(images_raw, save_path, plot_title, subtitiles = None):
    plt.clf()
    rows = np.sqrt(images_raw.shape[0]).astype(int)
    columns = np.ceil(images_raw.shape[0]/rows).astype(int)
    if rows == 1 and columns == 1:
        fig = plt.figure(figsize=(rows * 5/3 * IMAGE_SIZE, 
        (columns + 0.67) * IMAGE_SIZE))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(np.clip(images_raw[0]/np.amax(images_raw[0]), 0, 1))
        if subtitiles is not None:
            plt.title(subtitiles[0], fontsize=2.2 * SMALL_SIZE)
        else:
            plt.title(f"{1}-th raw image", fontsize=SMALL_SIZE)
    elif rows == 1 or columns == 1:
        fig, axes = plt.subplots(rows, columns,
        figsize=(2*IMAGE_SIZE, (columns + 1) * IMAGE_SIZE)
        )
        plt.setp(axes, xticks=[], yticks=[])
        # visualize query images
        for i in range(images_raw.shape[0]):
            axes[i].imshow(np.clip(images_raw[i]/np.amax(images_raw[i]), 0, 1))
            if subtitiles is not None:
                axes[i].set_title(subtitiles[i], fontsize=SMALL_SIZE)
            else:
                axes[i].set_title(f"{i}-th raw image", fontsize=SMALL_SIZE)
    else:
        fig, axes = plt.subplots(rows, columns, figsize=(rows * 5/3 * IMAGE_SIZE, 
        (columns + 0.67) * IMAGE_SIZE)
        )
        plt.setp(axes, xticks=[], yticks=[])
        # visualize query images
        for i in range(images_raw.shape[0]):
            axes[i // columns][i%columns].imshow(np.clip(images_raw[i]/np.amax(images_raw[i]), 0, 1))
            if subtitiles is not None:
                axes[i // columns][i%columns].set_title(subtitiles[i], fontsize=SMALL_SIZE)
            else:
                axes[i // columns][i%columns].set_title(f"{i}-th raw image", fontsize=SMALL_SIZE)
    plt.savefig(f"{save_path}", bbox_inches='tight')
    plt.close()

# plot subsampled images
def plot_subsample_images(images_raw, save_path, plot_title, subtitiles = None, max_num=2):
    show_indices = list(range(images_raw.shape[0]))
    if images_raw.shape[0]>max_num:
        show_indices = random.sample(range(images_raw.shape[0]), max_num)
        images_raw = images_raw[show_indices]
        if subtitiles is not None:
            subtitiles = [subtitiles[i] for i in show_indices]
    plot_images(images_raw, save_path, plot_title, subtitiles)
    return show_indices

# plot the optimized query
def plot_query(save_location):
    data = np.load(f"{save_location}/best.npz")
    images = data['best_params_x']
    plot_images(images, f"{save_location}/query.png", f"Optimized Queries in {images.shape[0]} trials")

def plot_target(save_location):
    data = np.load(f"{save_location}/datasets.npz")
    images = data['added_x']
    plot_images(images, f"{save_location}/differing.png", f"Differing Images in {images.shape[0]} trials")

def plot_checkpoint(save_location):
    file_list = sorted(os.listdir(save_location))
    loss_list, iter_list, norm_list, kern_sim_list = [], [], [], []
    parent_folder = '/'.join(save_location.split('/')[:-1])
    if os.path.exists(f"{parent_folder}/training_dynamics"):
        shutil.rmtree(f"{parent_folder}/training_dynamics")
    os.mkdir(f"{parent_folder}/training_dynamics")
    
    for file in file_list:
        if "checkpoint_" in file:
            checkpoint_idx = int(file.replace('checkpoint_', '').replace('.npz', ''))
            data = np.load(f"{save_location}/{file}")
            norm_list.append(float(np.linalg.norm(data['params_x'])))
            images = data['params_x']
            train_loss = data['train_loss']
            loss_list.append(float(train_loss))
            iter_list.append(checkpoint_idx)
            kern_sim_list.append(np.mean(data['kern_dist']))
            
    
    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    order = np.argsort(iter_list)
    # plot privacy loss of query for MSE optimization
    xs = np.array(iter_list)[order]
    ys = np.array(loss_list)[order]
    axes[0].plot(xs, - ys, label = f'Query (initial {-ys[0]:.3g})')
    axes[0].set_xlabel("iter")
    if save_location.find("kl")!=-1:
        axes[0].set_ylabel("KL divergence")
    else:
        axes[0].set_ylabel("Mean distance")
    data = np.load(f"{save_location}/best.npz")
    baseline_loss = data['base_loss']
    axes[0].plot([xs[0], xs[-1]], [- baseline_loss, - baseline_loss], label = f'Differ { - baseline_loss:.3g}')
    axes[0].legend(loc="lower right")
    # # plot norm of query for MSE optimization
    zs = np.array(norm_list)[order]
    axes[1].plot(xs, zs)
    axes[1].set_ylabel("query norm")
    axes[1].set_xlabel("iter")
    images = np.load(f"{save_location}/checkpoint_{str(list(np.array(iter_list)[order])[-1])}.npz")['params_x']
    axes[2].imshow(np.clip(images[0]/np.amax(images[0]), 0, 1))
    plt.savefig(f"{parent_folder}/training_dynamics/{save_location.split('/')[-1]}.png", bbox_inches='tight')
    plt.close()


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def plot_dist_AUC(save_location, time, num_samples, mean1, mean2, var1, var2, is_plotting = 1):
    # take matrix mean and covariance matrix, and plot lr distribution and attack AUC

    num_queries = mean1.shape[0]
    num_classes = mean1.shape[1]
    # reduce matrix to vector
    mean1, mean2 = mean1.T.reshape(-1), mean2.T.reshape(-1)
    var1, var2 = np.kron(np.eye(num_classes), var1), np.kron(np.eye(num_classes), var2)

    # compute AUC and plot fpr-tpr
    a = np.random.multivariate_normal(mean1, var1, num_samples)
    b = np.random.multivariate_normal(mean2, var2, num_samples)
    class_indicator = np.append(np.ones(num_samples), np.ones(num_samples) + 1)
    lr = multivariate_normal.pdf(np.append(a, b, axis = 0), mean1, var1)/ (multivariate_normal.pdf(np.append(a, b, axis = 0), mean2, var2) + 1e-30)
    fpr, tpr, _ = roc_curve(class_indicator, lr, pos_label = 1)
    if is_plotting == 1:
        plt.clf()
        plt.figure(figsize=(IMAGE_SIZE, IMAGE_SIZE))
        plt.semilogx()
        plt.semilogy()
        plt.plot(fpr, tpr, label = 'auc=%.3f' % auc(fpr, tpr))
        plt.xlim(1/num_samples, 1)
        plt.ylim(1/num_samples, 1)
        plt.xlabel("False Postive Rate")
        plt.ylabel("True Positive Rate")
        plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
        plt.legend(fontsize=8)
        plt.savefig(f"{save_location}/auc_at_time_{time}.png", bbox_inches='tight')
        plt.close()

        # plot histogram of lr ratio for samples from the two distributions
        plt.clf()
        fig=plt.figure(figsize=(2 * IMAGE_SIZE, IMAGE_SIZE))
        labels_temp = ['in', 'out']
        for i in range(2):
            fig.add_subplot(1, 2, i + 1)
            plt.grid()
            plt.xlabel('LR value')
            plt.ylabel('Number of samples')
            lr_temp = lr[i*num_samples: (i+1)*num_samples]
            plt.hist(lr_temp[~is_outlier(lr_temp)], bins = 20, label = f'{labels_temp[i]} predictions')
            plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
            plt.legend(fontsize=8)
        plt.title('LR histogram')
        plt.savefig(f"{save_location}/lr_at_time_{time}.png", bbox_inches='tight')
        plt.close()
    return auc(fpr, tpr)

def eval_reconstruction_error(differing_points, queries, save_location, name):
    fro_list, max_pix_diff_list, min_pix_diff_list, mean_pix_diff_list, pixel_range_list =[], [], [], [], []
    num_differ = differing_points.shape[0]
    for i_differ in range(num_differ):
        differ_image = np.clip(differing_points[i_differ]/np.amax(differing_points[i_differ]), 0, 1)
        fro_error, max_pix_diff, min_pix_diff, mean_pix_diff = np.inf, np.inf, np.inf, np.inf
        query_size = queries.shape[0]
        for j in range(query_size):
            query_image =  np.clip(queries[j]/np.amax(queries[j]), 0, 1)
            
            fro_error = min(fro_error, np.linalg.norm(differ_image - query_image))
            max_pix_diff, min_pix_diff, mean_pix_diff = min(np.max(np.abs(differ_image - query_image)), max_pix_diff), min(np.min(np.abs(differ_image - query_image)), min_pix_diff), min(np.mean(np.abs(differ_image - query_image)), mean_pix_diff)
            pixel_range = np.max(differ_image) - np.min(differ_image)
    
    fro_list.append(fro_error)
    max_pix_diff_list.append(max_pix_diff)
    min_pix_diff_list.append(min_pix_diff)
    mean_pix_diff_list.append(mean_pix_diff)
    pixel_range_list.append(pixel_range)
    
    return np.mean(np.array(fro_list))



def plot_one_folder(folder):
    runs_list = sorted(os.listdir(folder))
    differing_images, query_images, differing_classes, best_queries, best_kls, best_aucs, differ_kls, differ_aucs = [], [], [], [], [], [], [], []
    flag_image = False
    subtitles_differ, subtitles_query = [], []
    auc_gap_list = []
    for run_i in tqdm(runs_list):
        save_location = f"{folder}/{run_i}"
        if not os.path.exists(f"{save_location}/best.npz"):
            continue
        best_data = np.load(f"{save_location}/best.npz")
        if flag_image == False:
            differing_images = np.load(f"{save_location}/datasets.npz")['added_x']
            query_images = best_data['best_params_x']
            flag_image = True
        else:
            differing_images = np.append(differing_images, np.load(f"{save_location}/datasets.npz")['added_x'], axis=0)
            query_images = np.append(query_images, best_data['best_params_x'], axis=0)
        best_queries.append(best_data['best_params_x'])
        best_kls.append(best_data['best_train_loss'])
        best_aucs.append(best_data['best_auc'])
        differ_kls.append(best_data['base_loss'])
        differ_aucs.append(best_data['base_auc'])
        auc_gap_list.append(best_data['base_auc'] - best_data['best_auc'])
        subtitles_differ.append(f"Differing Data \n LOOD: {-best_data['base_loss']:.5f}\n AUC: {best_data['base_auc']:.5f}")
        subtitles_query.append(f"Optimized Query \n LOOD: {-best_data['best_train_loss']:.5f}\n AUC: {best_data['best_auc']:.5f}")
        plot_checkpoint(save_location)
        
    # log kl for differing images, and corresponding queries
    df = pd.DataFrame({'Differing Data': differ_kls, 'Optimized Query': best_kls})
    df.to_csv(f"{folder}/kl.csv")
    np.savez(f"{folder}/differ_images_and_query", differ = differing_images, query = query_images)

    # plot AUC gap
    plt.clf()
    fig = plt.figure(figsize=(4/3 * IMAGE_SIZE, 0.7 * IMAGE_SIZE))
    if (min(auc_gap_list)>=0):
        plt.hist(auc_gap_list, bins=np.arange(0, max(auc_gap_list) + 0.025, 0.025))
    else:
        plt.hist(auc_gap_list, bins=np.arange(min(auc_gap_list), max(auc_gap_list) + 0.025, 0.025))
    plt.xlabel("MIA performance gap for query and diff point in AUC score")
    plt.ylabel("Frequency")
    plt.savefig(f"{folder}/hist_auc_gap.png", bbox_inches='tight')

def log_one_folder(folder, name):
    differ_loss_list, differ_auc_list, differ_kl_list, differ_mse_list, differ_kl_grad_norm_list, differ_mse_grad_norm_list, differ_acc_list = [], [], [], [], [], [], []
    images_curve_high = None
    flag_high = False
    images_curve_low = None
    flag_low = False
    opt_loss_list, opt_auc_list = [], []
    k = 1

    differ_mean_kern_sim_list, differ_topk_kern_sim_list, differ_class_mean_kern_sim_list, differ_class_list = [], [], [], []
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    runs_list = sorted(os.listdir(folder))
    # print title of table
    specifications = name.split("-")
    content = ','.join([specifications[2*i+1] for i in range(len(specifications)//2)])
    
    for i in range(len(runs_list)):
        if i > len(runs_list) - 1:
            break
        run_i = runs_list[i]
        save_location = f"{folder}/{run_i}"
        if not os.path.exists(f"{save_location}/landscape_differ.npz"):
            runs_list.remove(run_i)
            continue
        data = np.load(f"{save_location}/best.npz")
        opt_loss, opt_auc, differ_auc, differ_loss, differ_acc = data['best_train_loss'], data['best_auc'], data['base_auc'], data['base_loss'], data['base_acc']
        differ_landscape = np.load(f"{save_location}/landscape_differ.npz")
        datasets = np.load(f"{save_location}/datasets.npz")
        differ_kl, differ_mse, differ_kl_grad_norm, differ_mse_grad_norm = differ_landscape['kl'], differ_landscape['mse'], differ_landscape['kl_grad_norm'], differ_landscape['mse_grad_norm']
        differ_kern_sim =  differ_landscape['kern_dist']
        added_x = datasets['added_x']
        added_y = np.argmax(datasets['added_y'])
        differ_class_list.append(added_y)
        # appending data
        differ_auc_list.append(float(differ_auc))
        differ_acc_list.append(float(differ_acc))
        if differ_auc >= 0.8:
            if flag_high == True:
                images_curve_high = np.append(images_curve_high, added_x, axis=0)
            else:
                images_curve_high = added_x
                flag_high = True
        else:
            if flag_low == True:
                images_curve_low = np.append(images_curve_low, added_x, axis=0)
            else:
                images_curve_low = added_x
                flag_low = True
        differ_loss_list.append(float(differ_loss))
        differ_kl_list.append(differ_kl)
        differ_mse_list.append(differ_mse)
        differ_kl_grad_norm_list.append(differ_kl_grad_norm)
        differ_mse_grad_norm_list.append(differ_mse_grad_norm)
        opt_auc_list.append(float(opt_auc))
        opt_loss_list.append(float(opt_loss))
        differ_mean_kern_sim_list.append(float(np.mean(differ_kern_sim)))
        ind = np.argpartition(differ_kern_sim, -k)[-k:]
        differ_topk_kern_sim_list.append(float(np.mean(differ_kern_sim[:,ind])))
        differ_class_mean_kern_sim_list.append(float(np.mean(differ_kern_sim[:, differ_kern_sim.shape[1]//10 * added_y : differ_kern_sim.shape[1]//10 * (added_y+1)])))
        
    if differ_loss_list!=[]:
        content = content + f', {np.array(opt_loss_list)}, {np.array(differ_loss_list)}, {np.array(opt_auc_list)}, {np.array(differ_auc_list)}, {(- np.array(opt_loss_list) + np.array(differ_loss_list))}, {(np.array(opt_auc_list) - np.array(differ_auc_list))}'

        # plot correlation between loss and auc
        df = pd.DataFrame(data={
            'auc': np.array(differ_auc_list),
            'mse': - np.array(differ_mse_list),
            'kl': - np.array(differ_kl_list),
            'kl_grad_norm': np.array(differ_kl_grad_norm_list),
            'mse_grad_norm': np.array(differ_mse_grad_norm_list),
            'mean_kern_sim': np.array(differ_mean_kern_sim_list),
            'max_kern_sim': np.array(differ_topk_kern_sim_list),
            'class_kern_sim': np.array(differ_class_mean_kern_sim_list),
            'acc': np.array(differ_acc_list)
        })
        
        df.to_csv(f"{folder}_auc_lood_kern_sim.csv")

        plt.clf()
        fig = plt.figure(figsize=(3,3))
        plt.scatter(np.array(differ_mean_kern_sim_list), np.array(differ_auc_list), alpha = 0.6, color = 'green')
        plt.xlabel("kernel similarity with dataset")
        plt.ylabel("auc on differing point")
        plt.savefig(f"{folder}_kern_sim_mean.png", bbox_inches='tight')
        plt.scatter(np.array(differ_topk_kern_sim_list), np.array(differ_auc_list), alpha = 0.6, color = 'green')
        plt.xlabel(f"kernel similarity with {k}-NN")
        plt.savefig(f"{folder}_kern_sim_topk.png", bbox_inches='tight')
        plt.scatter(np.array(differ_acc_list), np.array(differ_auc_list), alpha = 0.6, color = 'green')
        plt.xlabel(f"loss of differing point\n on leave-one-out dataset")
        plt.legend(fontsize=8)
        plt.savefig(f"{folder}_acc_auc.png", bbox_inches='tight')
        plt.close()
        
    # log the runs with the higest and lowest auc
    num_keep = 5
    temp_output_file = open(f"{folder}_vulnerable_points.txt" ,'w')
    temp_file_print = partial(double_print, output_file = temp_output_file)
    ind = np.argsort(differ_auc_list)
    temp_file_print(f"top  vulnerable {num_keep} points: {[runs_list[i] for i in ind[-num_keep:]]}")
    temp_file_print(f"auc scores: {[differ_auc_list[i] for i in ind[-num_keep:]]}")
    temp_file_print(f"indices: {ind[-num_keep:]}")
    temp_file_print(f"least vulnerable {num_keep} points: {[runs_list[i] for i in ind[:num_keep]]}")
    temp_file_print(f"auc scores: {[differ_auc_list[i] for i in ind[:num_keep]]}")
    temp_file_print(f"medium vulnerable {num_keep} points: {[runs_list[i] for i in ind[len(runs_list)//2: len(runs_list)//2 + num_keep]]}")
    temp_file_print(f"auc scores: {[differ_auc_list[i] for i in ind[len(runs_list)//2: len(runs_list)//2 + num_keep]]}")
    # restore np.load for future normal usage
    np.load = np_load_old
