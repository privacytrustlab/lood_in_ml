import os
from solver import Solver
from torch.backends import cudnn
from datetime import datetime
import json
from utils.param_parser import *
import time
import numpy as np

def save_config(config, shared_path):
    save_path = os.path.join(shared_path, "params.json")

    with open(save_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_time():
    return datetime.now().strftime("%d%m_%H%M%S")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='experiment_1')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_update_mode', type = str, default = 'epoch')
    parser.add_argument('--lr_schedule', action = DictParser, default = {'name': 'jump', 'start_v': 0.001, 'power' : 0.1, 'min_jump_pt': 25, 'jump_freq': 10
                                                                         })
    parser.add_argument('--valid_ratio', type=float, default=0)

    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--base_path', type=str, default='./models_in_relu_NTK/')  # where to save the experiment results
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--logs_path', type=str, default="logs")
    parser.add_argument('--validation_path', type=str, default="validation")

    parser.add_argument('--log_step', type=int, default=51)
    parser.add_argument('--save_every', type=int, default=51)
    parser.add_argument('--seed', type=int, default=time.time())
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--differ_data', type=int, default=None)


    parser.add_argument('--epsilon_test', type=float, default=0.01)
    parser.add_argument('--eps_iter_test', type=float, default=0.01)
    parser.add_argument('--nb_iter_test', type=int, default=1)
    config = parser.parse_args()

    config.model_path_ = config.model_path
    config.sample_path_ = config.sample_path
    config.logs_path_ = config.logs_path
    config.validation_path_ = config.validation_path

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    # training 100 models trained on classes 0 and 1 on cifar10 dataset
    for j in range(100):
        cudnn.benchmark = True

        config.time_now = get_time()
        config.id = f'experiment_{j}'

        shared_path = '{}{}'.format(config.base_path, config.id)

        config.model_path = "{}/{}".format(shared_path, config.model_path_)
        config.sample_path = "{}/{}".format(shared_path, config.sample_path_)
        config.logs_path = "{}/{}".format(shared_path, config.logs_path_)
        config.validation_path = "{}/{}".format(shared_path, config.validation_path_)

        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)

        if not os.path.exists(config.sample_path):
            os.makedirs(config.sample_path)

        if not os.path.exists(config.logs_path):
            os.makedirs(config.logs_path)

        if not os.path.exists(config.validation_path):
            os.makedirs(config.validation_path)

        config.seed = time.time()

        save_config(config, shared_path)
        print(config)
        solver = Solver(config)
        solver.train()