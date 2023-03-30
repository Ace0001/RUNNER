import os
import time

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import seed_everything
from tqdm import tqdm

from dataset import preprocess_adult_data
from model import Net
from utils import train_dp, evaluate_dp, evaluate_dp_new, evaluate_difference, mask_neuron_test
from utils import train_eo, evaluate_eo, evaluate_eo_new

def run_experiments(method='mixup', mode='dp', lam=0.5, num_exp=10, neuron_ratio=1):
    '''
    Retrain each model for 10 times and report the mean ap and dp.
    '''

    ap = []
    gap = []

    for i in range(num_exp):
        # i = 2
        seed_everything(i)
        print('On experiment', i)
        # get train/test data
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_adult_data(seed = i)

        # initialize model
        model = Net(input_size=len(X_train[0])).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        # run experiments
        ap_val_epoch = []
        gap_val_epoch = []
        ap_test_epoch = []
        gap_test_epoch = []
        our_start = time.time()
        for j in range(10):
            print('\nEpoch:', j)
            if mode == 'dp':
                train_dp(model, criterion, optimizer, X_train, A_train, y_train, method, lam, neuron_ratio)
                ap_val, gap_val = evaluate_dp_new(model, X_val, y_val, A_val)
                ap_test, gap_test = evaluate_dp_new(model, X_test, y_test, A_test)
                # evaluate_difference(model, X_test, y_test, A_test, optimizer=optimizer, mode='dp')
                # mask_neuron_test(model, X_test, y_test, A_test, criterion, optimizer, mode='dp')
                print('ap_test:', ap_test)
                print('gap_test:', gap_test)
            elif mode == 'eo':
                # mask_neuron_test(model, X_test, y_test, A_test, criterion, optimizer=optimizer, mode='eo')
                train_eo(model, criterion, optimizer, X_train, A_train, y_train, method, lam, neuron_ratio)
                ap_val, gap_val = evaluate_eo_new(model, X_val, y_val, A_val)
                ap_test, gap_test = evaluate_eo_new(model, X_test, y_test, A_test)
                # evaluate_difference(model, X_test, y_test, A_test, optimizer=optimizer, mode='eo')
                # mask_neuron_test(model, X_test, y_test, A_test, criterion, optimizer, mode='eo')


                print('ap_test:', ap_test)
                print('gap_test:', gap_test)
            if j > 0:
                ap_val_epoch.append(ap_val)
                ap_test_epoch.append(ap_test)
                gap_val_epoch.append(gap_val)
                gap_test_epoch.append(gap_test)

        # best model based on val performance
        idx = gap_val_epoch.index(min(gap_val_epoch))
        gap.append(gap_test_epoch[idx])
        ap.append(ap_test_epoch[idx])
        print('--------INDEX---------')
        print('idx: ', idx+1)
        print('ap_test:', ap_test_epoch[idx])
        print('gap_test:', gap_test_epoch[idx])
        our_end = time.time()
        print('time costs:{} s'.format(our_end - our_start))


    print('--------AVG---------')
    print('Average Precision: %.6f' % np.mean(ap))
    print(mode + ' gap: %.6f' % np.mean(gap))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adult Experiment')
    parser.add_argument('--method', default='mixup', type=str, help='mixup/GapReg/erm')
    parser.add_argument('--mode', default='dp', type=str, help='dp/eo')
    parser.add_argument('--lam', default=1, type=float, help='Lambda for regularization')
    parser.add_argument('--neuron_ratio', default=1, type=float, help='% of topk importance neuron')
    parser.add_argument('--ex_num', default=10, type=int, help='num of experiment')
    args = parser.parse_args()

    run_experiments(args.method, args.mode, args.lam, num_exp=args.ex_num, neuron_ratio=args.neuron_ratio)

