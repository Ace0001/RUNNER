import copy
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
import torch
import numpy as np
from numpy.random import beta
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sample_batch_sen_id(X, A, y, batch_size):
    batch_idx = np.random.choice([i for i in range(len(A))], size=batch_size*4, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y

def sample_batch_sen_idx(X, A, y, batch_size, s):
    batch_idx = np.random.choice(np.where(A == s)[0], size=batch_size, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y


def sample_batch_sen_idx_y(X, A, y, batch_size, s):
    batch_idx = []
    for i in range(2):
        idx = list(set(np.where(A == s)[0]) & set(np.where(y == i)[0]))
        batch_idx += np.random.choice(idx, size=batch_size, replace=False).tolist()

    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y


def all_sen_idx_y(X, A, y, s, i):
    all_id = list(set(np.where(A == s)[0]) & set(np.where(y == i)[0]))
    batch_x = X[all_id]
    batch_y = y[all_id]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y

def train_dp(model, criterion, optimizer, X_train, A_train, y_train, method, lam, neuron_ratio, batch_size=500, niter=100):
    model.train()
    # sum1_0, sum2_0, sum3_0 = 0, 0, 0
    # sum1_1, sum2_1, sum3_1 = 0, 0, 0

    for it in range(niter):

        # Gender Split
        batch_x_0, batch_y_0 = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 0)
        batch_x_1, batch_y_1 = sample_batch_sen_idx(X_train, A_train, y_train, batch_size, 1)

        if method == 'mixup':
            # Fair Mixup
            alpha = 1
            gamma = beta(alpha, alpha)

            batch_x_mix = batch_x_0 * gamma + batch_x_1 * (1 - gamma)
            batch_x_mix = batch_x_mix.requires_grad_(True)

            output, _, _, _ = model(batch_x_mix)

            # gradient regularization
            gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]

            batch_x_d = batch_x_1 - batch_x_0
            grad_inn = (gradx * batch_x_d).sum(1)
            E_grad = grad_inn.mean(0)
            loss_reg = torch.abs(E_grad)

        elif method == 'GapReg':
            # Gap Regularizatioon
            output_0, _, _, _ = model(batch_x_0)
            output_1, _, _, _ = model(batch_x_1)
            loss_reg = torch.abs(output_0.mean() - output_1.mean())
        elif method == 'fairlearn':
            batch_x = torch.cat((batch_x_0, batch_x_1), 0)
            batch_y = torch.cat((batch_y_0, batch_y_1), 0).cpu().numpy()
            batch_A = torch.cat((torch.zeros(len(batch_x_0)), torch.ones(len(batch_x_1))), 0).cpu().numpy()

            output, _, _, _ = model(batch_x)
            pred = np.int64(output.cpu().detach().numpy() > 0.5)
            loss_reg = demographic_parity_difference(batch_y, pred, sensitive_features=batch_A)
        elif method == "NeuronImportance_GapReg":
            important_index1, important_index2, _, _ = cal_importance_gapReg(model, optimizer, batch_x_0, batch_x_1, neuron_ratio, mode='dp')

            output_0, add1_0, add2_0, add3_0 = model(batch_x_0)
            output_1, add1_1, add2_1, add3_1 = model(batch_x_1)
            loss_reg = torch.abs(add1_0[important_index1] - add1_1[important_index1]).mean() + \
                        torch.abs(add2_0[important_index2] - add2_1[important_index2]).mean()
            # loss_reg = torch.abs(add1_0[important_index1] - add1_1[important_index1]).mean() + \
            #             torch.abs(add2_0[important_index2] - add2_1[important_index2]).mean() + \
            #             torch.abs(add3_0 - add3_1).mean()
            if it % 100 == 0:
                print('loss_reg:', loss_reg.item())
        else:
            # ERM
            loss_reg = 0

        # ERM loss
        batch_x = torch.cat((batch_x_0, batch_x_1), 0)
        batch_y = torch.cat((batch_y_0, batch_y_1), 0)

        output, _, _, _ = model(batch_x)
        loss_sup = criterion(output, batch_y)

        # final loss
        loss = loss_sup + lam * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print('\ndif1: ', abs(sum1_0 - sum1_1) / niter)
    # print('dif2: ', abs(sum2_0 - sum2_1) / niter)
    # print('dif3: ', abs(sum3_0 - sum3_1) / niter)


def train_eo(model, criterion, optimizer, X_train, A_train, y_train, method, lam, neuron_ratio, batch_size=500, niter=100):
    model.train()
    for it in range(niter):

        # Gender Split
        batch_x_0, batch_y_0 = sample_batch_sen_idx_y(X_train, A_train, y_train, batch_size, 0)
        batch_x_1, batch_y_1 = sample_batch_sen_idx_y(X_train, A_train, y_train, batch_size, 1)

        # separate class
        batch_x_0_ = [batch_x_0[:batch_size], batch_x_0[batch_size:]]
        batch_x_1_ = [batch_x_1[:batch_size], batch_x_1[batch_size:]]

        # ERM loss
        batch_x = torch.cat((batch_x_0, batch_x_1), 0)
        batch_y = torch.cat((batch_y_0, batch_y_1), 0)

        if method == 'mixup':
            loss_reg = 0
            alpha = 1
            for i in range(2):
                gamma = beta(alpha, alpha)
                batch_x_0_i = batch_x_0_[i]
                batch_x_1_i = batch_x_1_[i]

                batch_x_mix = batch_x_0_i * gamma + batch_x_1_i * (1 - gamma)
                batch_x_mix = batch_x_mix.requires_grad_(True)
                output, _, _, _ = model(batch_x_mix)

                # gradient regularization
                gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]
                batch_x_d = batch_x_1_i - batch_x_0_i
                grad_inn = (gradx * batch_x_d).sum(1)
                loss_reg += torch.abs(grad_inn.mean())

        elif method == "GapReg":
            loss_reg = 0
            for i in range(2):
                batch_x_0_i = batch_x_0_[i]
                batch_x_1_i = batch_x_1_[i]

                output_0, _, _, _ = model(batch_x_0_i)
                output_1, _, _, _ = model(batch_x_1_i)
                loss_reg += torch.abs(output_0.mean() - output_1.mean())
        elif method == "NeuronImportance":
            loss_reg = 0
            important_index1, important_index2, _, _ = cal_importance(model, optimizer, batch_x, batch_y, neuron_ratio)
            for i in range(2):
                batch_x_0_i = batch_x_0_[i]
                batch_x_1_i = batch_x_1_[i]

                output_0, add1_0, add2_0, add3_0 = model(batch_x_0_i)
                output_1, add1_1, add2_1, add3_1 = model(batch_x_1_i)
                loss_reg += torch.abs(add1_0[important_index1] - add1_1[important_index1]).mean() + \
                            torch.abs(add2_0[important_index2] - add2_1[important_index2]).mean() + \
                            torch.abs(add3_0 - add3_1).mean()
        elif method == "NeuronImportance_GapReg":
            loss_reg = 0
            loss_reg0 = 0
            important_index1, important_index2, _, _ = cal_importance_gapReg(model, optimizer, batch_x_0_, batch_x_1_, neuron_ratio, mode='eo')
            for i in range(2):
                batch_x_0_i = batch_x_0_[i]
                batch_x_1_i = batch_x_1_[i]

                output_0, add1_0, add2_0, add3_0 = model(batch_x_0_i)
                output_1, add1_1, add2_1, add3_1 = model(batch_x_1_i)
                # loss_reg += torch.abs(add1_0[important_index1] - add1_1[important_index1]).mean() + \
                #             torch.abs(add2_0[important_index2] - add2_1[important_index2]).mean()
                loss_reg += torch.abs(add1_0[important_index1] - add1_1[important_index1]).mean() + \
                            torch.abs(add2_0[important_index2] - add2_1[important_index2]).mean() + \
                            torch.abs(add3_0 - add3_1).mean()

                loss_reg0 += torch.abs(output_0.mean() - output_1.mean())
            if it % 100 == 0:
                print('loss_reg0:', loss_reg0.item())
                print('loss_reg:', loss_reg.item())
        elif method == 'fairlearn':
            batch_y_array = batch_y.cpu().numpy()
            batch_A = torch.cat((torch.zeros(len(batch_x_0)), torch.ones(len(batch_x_1))), 0).cpu().numpy()

            output, _, _, _ = model(batch_x)
            pred = np.int64(output.cpu().detach().numpy() > 0.5)
            loss_reg = equalized_odds_difference(batch_y_array, pred, sensitive_features=batch_A)
        else:
            # ERM
            loss_reg = 0

        if method == "van":
            batch_x, batch_y = sample_batch_sen_id(X_train, A_train, y_train, batch_size)
        output, _, _, _ = model(batch_x)
        loss_sup = criterion(output, batch_y)
        if it % 100 == 0:
            print('loss_sup:', loss_sup.item())

        # final loss
        loss = loss_sup + lam * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_dp(model, X_test, y_test, A_test):
    model.eval()

    # calculate DP gap
    idx_0 = np.where(A_test == 0)[0]
    idx_1 = np.where(A_test == 1)[0]

    X_test_0 = X_test[idx_0]
    X_test_1 = X_test[idx_1]
    X_test_0 = torch.tensor(X_test_0).cuda().float()
    X_test_1 = torch.tensor(X_test_1).cuda().float()

    pred_0, _, _, _ = model(X_test_0)
    pred_1, _, _, _ = model(X_test_1)

    gap = pred_0.mean() - pred_1.mean()
    gap = abs(gap.data.cpu().numpy())

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    output, _, _, _ = model(X_test_cuda)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    return ap, gap


def evaluate_eo(model, X_test, y_test, A_test, testing=False, optimizer=None):
    model.eval()
    idx_00 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 0)[0]))
    idx_01 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 1)[0]))
    idx_10 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 0)[0]))
    idx_11 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 1)[0]))

    X_test_00 = X_test[idx_00]
    X_test_01 = X_test[idx_01]
    X_test_10 = X_test[idx_10]
    X_test_11 = X_test[idx_11]

    X_test_00 = torch.tensor(X_test_00).cuda().float()
    X_test_01 = torch.tensor(X_test_01).cuda().float()
    X_test_10 = torch.tensor(X_test_10).cuda().float()
    X_test_11 = torch.tensor(X_test_11).cuda().float()

    if testing:
        # diff
        X_test_0 = torch.cat((X_test_00, X_test_01), dim=0)
        X_test_1 = torch.cat((X_test_10, X_test_11), dim=0)
        sum1_0, sum2_0, sum3_0 = model.neuron_sum(X_test_0)
        sum1_1, sum2_1, sum3_1 = model.neuron_sum(X_test_1)

        sum1_00, sum2_00, sum3_00 = model.neuron_sum(X_test_00)
        sum1_01, sum2_01, sum3_01 = model.neuron_sum(X_test_01)
        sum1_10, sum2_10, sum3_10 = model.neuron_sum(X_test_10)
        sum1_11, sum2_11, sum3_11 = model.neuron_sum(X_test_11)

        # difference
        dif1 = sum1_0 - sum1_1
        dif2 = sum2_0 - sum2_1
        dif3 = sum3_0 - sum3_1

        dif1_0 = sum1_00 - sum1_10
        dif2_0 = sum2_00 - sum2_10
        dif3_0 = sum3_00 - sum3_10

        dif1_1 = sum1_01 - sum1_11
        dif2_1 = sum2_01 - sum2_11
        dif3_1 = sum3_01 - sum3_11

        # difference abs
        dif1_abs = abs(sum1_0 - sum1_1)
        dif2_abs = abs(sum2_0 - sum2_1)
        dif3_abs = abs(sum3_0 - sum3_1)

        dif1_0_abs = abs(sum1_00 - sum1_10)
        dif2_0_abs = abs(sum2_00 - sum2_10)
        dif3_0_abs = abs(sum3_00 - sum3_10)

        dif1_1_abs = abs(sum1_01 - sum1_11)
        dif2_1_abs = abs(sum2_01 - sum2_11)
        dif3_1_abs = abs(sum3_01 - sum3_11)

        # topk
        value1, index1 = dif1_abs.topk(10, dim=0, largest=True)
        value2, index2 = dif2_abs.topk(10, dim=0, largest=True)

        value1_0, index1_0 = dif1_0_abs.topk(10, dim=0, largest=True)
        value2_0, index2_0 = dif2_0_abs.topk(10, dim=0, largest=True)

        value1_1, index1_1 = dif1_1_abs.topk(10, dim=0, largest=True)
        value2_1, index2_1 = dif2_1_abs.topk(10, dim=0, largest=True)

        torch.set_printoptions(precision=4, sci_mode=False)
        # print('\ndif1: ', dif1)
        # print('dif2: ', dif2)
        # print('dif3: ', dif3)
        #
        # print('\ndif1_0: ', dif1_0)
        # print('dif2_0: ', dif2_0)
        # print('dif3_0: ', dif3_0)
        #
        # print('\ndif1_1: ', dif1_1)
        # print('dif2_1: ', dif2_1)
        # print('dif3_1: ', dif3_1)
        #
        # print('\ndif1_abs: ', dif1_abs)
        # print('dif2_abs: ', dif2_abs)
        # print('dif3_abs: ', dif3_abs)
        #
        # print('\ndif1_0_abs: ', dif1_0_abs)
        # print('dif2_0_abs: ', dif2_0_abs)
        # print('dif3_0_abs: ', dif3_0_abs)
        #
        # print('\ndif1_1_abs: ', dif1_1_abs)
        # print('dif2_1_abs: ', dif2_1_abs)
        # print('dif3_1_abs: ', dif3_1_abs)
        #
        # print('\nindex1: ', index1)
        # print('value1: ', value1)
        # print('index2: ', index2)
        # print('value2: ', value2)
        #
        # print('\nindex1_0: ', index1_0)
        # print('value1_0: ', value1_0)
        # print('index2_0: ', index2_0)
        # print('value2_0: ', value2_0)
        #
        # print('\nindex1_1: ', index1_1)
        # print('value1_1: ', value1_1)
        # print('index2_1: ', index2_1)
        # print('value2_1: ', value2_1)

        # importance
        X_test_T = torch.tensor(X_test).cuda().float()
        y_test_T = torch.tensor(y_test).cuda().float()

        important_index1, important_index2, important_value1, important_value2 = cal_importance(model, optimizer,
                                                                                                X_test_T, y_test_T)
        # print('\nimportant_index1: ', important_index1)
        # print('important_value1: ', important_value1)
        # print('important_index2: ', important_index2)
        # print('important_value2: ', important_value2)
        #
        # important_index1_00, important_index2_00, important_value1_00, important_value2_00 = cal_importance(model, optimizer, X_test_00, y_test_T[idx_00])
        # print('\nimportant_index1_00: ', important_index1_00)
        # print('important_value1_00: ', important_value1_00)
        # print('important_index2_00: ', important_index2_00)
        # print('important_value2_00: ', important_value2_00)
        # important_index1_01, important_index2_01, important_value1_01, important_value2_01 = cal_importance(model, optimizer, X_test_01, y_test_T[idx_01])
        # print('\nimportant_index1_01: ', important_index1_01)
        # print('important_value1_01: ', important_value1_01)
        # print('important_index2_01: ', important_index2_01)
        # print('important_value2_01: ', important_value2_01)
        # important_index1_10, important_index2_10, important_value1_10, important_value2_10 = cal_importance(model, optimizer, X_test_10, y_test_T[idx_10])
        # print('\nimportant_index1_10: ', important_index1_10)
        # print('important_value1_10: ', important_value1_10)
        # print('important_index2_10: ', important_index2_10)
        # print('important_value2_10: ', important_value2_10)
        # important_index1_11, important_index2_11, important_value1_11, important_value2_11 = cal_importance(model, optimizer, X_test_11, y_test_T[idx_11])
        # print('\nimportant_index1_11: ', important_index1_11)
        # print('important_value1_11: ', important_value1_11)
        # print('important_index2_11: ', important_index2_11)
        # print('important_value2_11: ', important_value2_11)

        # print('\nimportant_index1: ', important_index1)
        # print('important_index2: ', important_index2)
        # print('\nimportant_diff_value1: ', dif1_abs[important_index1])
        # print('important_avg_diff_value1: ', dif1_abs[important_index1].mean())
        # print('\nimportant_diff_value2: ', dif2_abs[important_index2])
        # print('important_avg_diff_value2: ', dif2_abs[important_index2].mean())
        # print('\nimportant_diff_value1_0: ', dif1_0_abs[important_index1])
        # print('important_avg_diff_value1_0: ', dif1_0_abs[important_index1].mean())
        # print('\nimportant_diff_value2_0: ', dif2_0_abs[important_index2])
        # print('important_avg_diff_value2_0: ', dif2_0_abs[important_index2].mean())
        # print('\nimportant_diff_value1_1: ', dif1_1_abs[important_index1])
        # print('important_avg_diff_value1_1: ', dif1_1_abs[important_index1].mean())
        # print('\nimportant_diff_value2_1: ', dif2_1_abs[important_index2])
        # print('important_avg_diff_value2_1: ', dif2_1_abs[important_index2].mean())

        print('\ntop20%')
        print('\nimportant_diff_value1: ', dif1_abs[important_index1[0:40]])
        print('important_avg_diff_value1: ', dif1_abs[important_index1[0:40]].mean())
        print('\ntop20%-40%')
        print('\nimportant_diff_value1: ', dif1_abs[important_index1[40:80]])
        print('important_avg_diff_value1: ', dif1_abs[important_index1[40:80]].mean())
        print('\ntop40%-60%')
        print('\nimportant_diff_value1: ', dif1_abs[important_index1[80:120]])
        print('important_avg_diff_value1: ', dif1_abs[important_index1[80:120]].mean())
        print('\ntop60%-80%')
        print('\nimportant_diff_value1: ', dif1_abs[important_index1[120:160]])
        print('important_avg_diff_value1: ', dif1_abs[important_index1[120:160]].mean())
        print('\ntop80%-100%')
        print('\nimportant_diff_value1: ', dif1_abs[important_index1[160:200]])
        print('important_avg_diff_value1: ', dif1_abs[important_index1[160:200]].mean())

        print('\ntop20%')
        print('\nimportant_diff_value2: ', dif2_abs[important_index2[0:40]])
        print('important_avg_diff_value2: ', dif2_abs[important_index2[0:40]].mean())
        print('\ntop20%-40%')
        print('\nimportant_diff_value2: ', dif2_abs[important_index2[40:80]])
        print('important_avg_diff_value2: ', dif2_abs[important_index2[40:80]].mean())
        print('\ntop40%-60%')
        print('\nimportant_diff_value2: ', dif2_abs[important_index2[80:120]])
        print('important_avg_diff_value2: ', dif2_abs[important_index2[80:120]].mean())
        print('\ntop60%-80%')
        print('\nimportant_diff_value2: ', dif2_abs[important_index2[120:160]])
        print('important_avg_diff_value2: ', dif2_abs[important_index2[120:160]].mean())
        print('\ntop80%-100%')
        print('\nimportant_diff_value2: ', dif2_abs[important_index2[160:200]])
        print('important_avg_diff_value2: ', dif2_abs[important_index2[160:200]].mean())

        print('\ntop20%')
        print('\nimportant_diff_value1_0: ', dif1_0_abs[important_index1[0:40]])
        print('important_avg_diff_value1_0: ', dif1_0_abs[important_index1[0:40]].mean())
        print('\ntop20%-40%')
        print('\nimportant_diff_value1_0: ', dif1_0_abs[important_index1[40:80]])
        print('important_avg_diff_value1_0: ', dif1_0_abs[important_index1[40:80]].mean())
        print('\ntop40%-60%')
        print('\nimportant_diff_value1_0: ', dif1_0_abs[important_index1[80:120]])
        print('important_avg_diff_value1_0: ', dif1_0_abs[important_index1[80:120]].mean())
        print('\ntop60%-80%')
        print('\nimportant_diff_value1_0: ', dif1_0_abs[important_index1[120:160]])
        print('important_avg_diff_value1_0: ', dif1_0_abs[important_index1[120:160]].mean())
        print('\ntop80%-100%')
        print('\nimportant_diff_value1_0: ', dif1_0_abs[important_index1[160:200]])
        print('important_avg_diff_value1_0: ', dif1_0_abs[important_index1[160:200]].mean())

        print('\ntop20%')
        print('\nimportant_diff_value2_0: ', dif2_0_abs[important_index2[0:40]])
        print('important_avg_diff_value2_0: ', dif2_0_abs[important_index2[0:40]].mean())
        print('\ntop20%-40%')
        print('\nimportant_diff_value2_0: ', dif2_0_abs[important_index2[40:80]])
        print('important_avg_diff_value2_0: ', dif2_0_abs[important_index2[40:80]].mean())
        print('\ntop40%-60%')
        print('\nimportant_diff_value2_0: ', dif2_0_abs[important_index2[80:120]])
        print('important_avg_diff_value2_0: ', dif2_0_abs[important_index2[80:120]].mean())
        print('\ntop60%-80%')
        print('\nimportant_diff_value2_0: ', dif2_0_abs[important_index2[120:160]])
        print('important_avg_diff_value2_0: ', dif2_0_abs[important_index2[120:160]].mean())
        print('\ntop80%-100%')
        print('\nimportant_diff_value2_0: ', dif2_0_abs[important_index2[160:200]])
        print('important_avg_diff_value2_0: ', dif2_0_abs[important_index2[160:200]].mean())

        print('\ntop20%')
        print('\nimportant_diff_value1_1: ', dif1_1_abs[important_index1[0:40]])
        print('important_avg_diff_value1_1: ', dif1_1_abs[important_index1[0:40]].mean())
        print('\ntop20%-40%')
        print('\nimportant_diff_value1_1: ', dif1_1_abs[important_index1[40:80]])
        print('important_avg_diff_value1_1: ', dif1_1_abs[important_index1[40:80]].mean())
        print('\ntop40%-60%')
        print('\nimportant_diff_value1_1: ', dif1_1_abs[important_index1[80:120]])
        print('important_avg_diff_value1_1: ', dif1_1_abs[important_index1[80:120]].mean())
        print('\ntop60%-80%')
        print('\nimportant_diff_value1_1: ', dif1_1_abs[important_index1[120:160]])
        print('important_avg_diff_value1_1: ', dif1_1_abs[important_index1[120:160]].mean())
        print('\ntop80%-100%')
        print('\nimportant_diff_value1_1: ', dif1_1_abs[important_index1[160:200]])
        print('important_avg_diff_value1_1: ', dif1_1_abs[important_index1[160:200]].mean())

        print('\ntop20%')
        print('\nimportant_diff_value2_1: ', dif2_1_abs[important_index2[0:40]])
        print('important_avg_diff_value2_1: ', dif2_1_abs[important_index2[0:40]].mean())
        print('\ntop20%-40%')
        print('\nimportant_diff_value2_1: ', dif2_1_abs[important_index2[40:80]])
        print('important_avg_diff_value2_1: ', dif2_1_abs[important_index2[40:80]].mean())
        print('\ntop40%-60%')
        print('\nimportant_diff_value2_1: ', dif2_1_abs[important_index2[80:120]])
        print('important_avg_diff_value2_1: ', dif2_1_abs[important_index2[80:120]].mean())
        print('\ntop60%-80%')
        print('\nimportant_diff_value2_1: ', dif2_1_abs[important_index2[120:160]])
        print('important_avg_diff_value2_1: ', dif2_1_abs[important_index2[120:160]].mean())
        print('\ntop80%-100%')
        print('\nimportant_diff_value2_1: ', dif2_1_abs[important_index2[160:200]])
        print('important_avg_diff_value2_1: ', dif2_1_abs[important_index2[160:200]].mean())

        important_index1_0, important_index2_0, important_value1_0, important_value2_0 = cal_importance(model,
                                                                                                        optimizer,
                                                                                                        torch.cat((
                                                                                                            X_test_00,
                                                                                                            X_test_10),
                                                                                                            dim=0),
                                                                                                        torch.cat((
                                                                                                            y_test_T[
                                                                                                                idx_00],
                                                                                                            y_test_T[
                                                                                                                idx_10]),
                                                                                                            dim=0))
        important_index1_1, important_index2_1, important_value1_1, important_value2_1 = cal_importance(model,
                                                                                                        optimizer,
                                                                                                        torch.cat((
                                                                                                            X_test_01,
                                                                                                            X_test_11),
                                                                                                            dim=0),
                                                                                                        torch.cat((
                                                                                                            y_test_T[
                                                                                                                idx_01],
                                                                                                            y_test_T[
                                                                                                                idx_11]),
                                                                                                            dim=0))

        # print('\nimportant_index1: ', important_index1)
        # print('important_diff_value1: ', dif1_abs[important_index1])
        # print('\nimportant_index2: ', important_index2)
        # print('important_diff_value2: ', dif2_abs[important_index2])
        #
        # print('\nimportant_index1_0: ', important_index1_0)
        # print('important_diff_value1_0: ', dif1_0_abs[important_index1_0])
        # print('\nimportant_index2_0: ', important_index2_0)
        # print('important_diff_value2_0: ', dif2_0_abs[important_index2_0])
        #
        # print('\nimportant_index1_1: ', important_index1_1)
        # print('important_diff_value1_1: ', dif1_1_abs[important_index1_1])
        # print('\nimportant_index2_1: ', important_index2_1)
        # print('important_diff_value2_1: ', dif2_1_abs[important_index2_1])

    pred_00, _, _, _ = model(X_test_00)
    pred_01, _, _, _ = model(X_test_01)
    pred_10, _, _, _ = model(X_test_10)
    pred_11, _, _, _ = model(X_test_11)

    gap_0 = pred_00.mean() - pred_10.mean()
    gap_1 = pred_01.mean() - pred_11.mean()
    gap_0 = abs(gap_0.data.cpu().numpy())
    gap_1 = abs(gap_1.data.cpu().numpy())

    gap = gap_0 + gap_1

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    output, _, _, _ = model(X_test_cuda)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    return ap, gap


def cal_importance(model, optimizer, x_train, y_train, neuron_ratio):
    model_cal = copy.deepcopy(model)
    optimizer_cal = copy.deepcopy(optimizer)
    criterion = nn.BCELoss()

    model_cal.train()
    output, _, _, _ = model_cal(x_train)
    loss = criterion(output, y_train)

    optimizer_cal.zero_grad()
    loss.backward(retain_graph=True)

    for name, param in model_cal.named_parameters():
        if name == 'fc1.weight':
            layer1 = param
        elif name == 'fc2.weight':
            layer2 = param

    nunits1 = layer1.shape[0]
    nunits2 = layer2.shape[0]

    criteria_layer1 = (layer1 * layer1.grad).data.pow(2).view(nunits1, -1).sum(dim=1)
    criteria_layer2 = (layer2 * layer2.grad).data.pow(2).view(nunits2, -1).sum(dim=1)

    # print('criteria_layer1:', criteria_layer1)
    # print('criteria_layer2:', criteria_layer2)

    value1, index1 = criteria_layer1.topk(int(neuron_ratio * nunits1), dim=0, largest=True)
    value2, index2 = criteria_layer2.topk(int(neuron_ratio * nunits2), dim=0, largest=True)

    return index1, index2, value1, value2


def cal_importance_gapReg(model, optimizer, batch_x_0_, batch_x_1_, neuron_ratio, mode=None):
    model_cal = copy.deepcopy(model)
    optimizer_cal = copy.deepcopy(optimizer)
    criterion = nn.BCELoss()

    model_cal.train()

    loss_reg = 0
    if mode == 'eo':
        for i in range(2):
            batch_x_0_i = batch_x_0_[i]
            batch_x_1_i = batch_x_1_[i]

            output_0, _, _, _ = model_cal(batch_x_0_i)
            output_1, _, _, _ = model_cal(batch_x_1_i)
            loss_reg += torch.abs(output_0.mean() - output_1.mean())
    elif mode == 'dp':
        output_0, _, _, _ = model_cal(batch_x_0_)
        output_1, _, _, _ = model_cal(batch_x_1_)
        loss_reg = torch.abs(output_0.mean() - output_1.mean())
    else:
        print("Error: No mode in cal_importance_gapReg()")
    loss = loss_reg

    optimizer_cal.zero_grad()
    loss.backward(retain_graph=True)

    for name, param in model_cal.named_parameters():
        if name == 'fc1.weight':
            layer1 = param
        elif name == 'fc2.weight':
            layer2 = param

    nunits1 = layer1.shape[0]
    nunits2 = layer2.shape[0]

    criteria_layer1 = (layer1 * layer1.grad).data.pow(2).view(nunits1, -1).sum(dim=1)
    criteria_layer2 = (layer2 * layer2.grad).data.pow(2).view(nunits2, -1).sum(dim=1)

    value1, index1 = criteria_layer1.topk(int(neuron_ratio * nunits1), dim=0, largest=True)
    value2, index2 = criteria_layer2.topk(int(neuron_ratio * nunits2), dim=0, largest=True)

    return index1, index2, value1, value2


def evaluate_dp_new(model, X_test, y_test, A_test):
    model.eval()

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    output, _, _, _ = model(X_test_cuda)
    pred = np.int64(output.cpu().detach().numpy() > 0.5)
    dp = demographic_parity_difference(y_test, pred, sensitive_features=A_test)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    return ap, dp


def evaluate_eo_new(model, X_test, y_test, A_test, testing=False, optimizer=None):
    model.eval()

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    output, _, _, _ = model(X_test_cuda)
    pred = np.int64(output.cpu().detach().numpy() > 0.5)
    eo = equalized_odds_difference(y_test, pred, sensitive_features=A_test)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    return ap, eo


def evaluate_difference(model, X_test, y_test, A_test, optimizer=None, mode=None):
    model.eval()
    optimizer_cal = copy.deepcopy(optimizer)
    idx_00 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 0)[0]))
    idx_01 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 1)[0]))
    idx_10 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 0)[0]))
    idx_11 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 1)[0]))

    X_test_00 = X_test[idx_00]
    X_test_01 = X_test[idx_01]
    X_test_10 = X_test[idx_10]
    X_test_11 = X_test[idx_11]

    X_test_00 = torch.tensor(X_test_00).cuda().float()
    X_test_01 = torch.tensor(X_test_01).cuda().float()
    X_test_10 = torch.tensor(X_test_10).cuda().float()
    X_test_11 = torch.tensor(X_test_11).cuda().float()

    X_test = torch.tensor(X_test).cuda().float()
    y_test = torch.tensor(y_test).cuda().float()

    # diff
    X_test_0 = torch.cat((X_test_00, X_test_01), dim=0)
    X_test_1 = torch.cat((X_test_10, X_test_11), dim=0)
    outputs_0, sum1_0, sum2_0, sum3_0 = model(X_test_0)
    outputs_1, sum1_1, sum2_1, sum3_1 = model(X_test_1)

    outputs_00, sum1_00, sum2_00, sum3_00 = model(X_test_00)
    outputs_01, sum1_01, sum2_01, sum3_01 = model(X_test_01)
    outputs_10, sum1_10, sum2_10, sum3_10 = model(X_test_10)
    outputs_11, sum1_11, sum2_11, sum3_11 = model(X_test_11)

    # difference abs
    dif1_abs = abs(sum1_0 - sum1_1)
    dif2_abs = abs(sum2_0 - sum2_1)
    dif3_abs = abs(sum3_0 - sum3_1)

    dif1_0_abs = abs(sum1_00 - sum1_10)
    dif2_0_abs = abs(sum2_00 - sum2_10)
    dif3_0_abs = abs(sum3_00 - sum3_10)

    dif1_1_abs = abs(sum1_01 - sum1_11)
    dif2_1_abs = abs(sum2_01 - sum2_11)
    dif3_1_abs = abs(sum3_01 - sum3_11)

    torch.set_printoptions(precision=10, sci_mode=False)

    # importance

    # separate class
    batch_x_0_ = [X_test_00, X_test_01]
    batch_x_1_ = [X_test_10, X_test_11]

    if mode == 'eo':
        important_index1_Reg, important_index2_Reg, important_value1_Reg, important_value2_Reg = cal_importance_gapReg(
            model, optimizer_cal, batch_x_0_, batch_x_1_, 1, mode='eo')
        important_index1_CE, important_index2_CE, important_value1_CE, important_value2_CE = cal_importance(
            model, optimizer_cal, X_test, y_test, 1)
        dif1_abs = dif1_0_abs + dif1_1_abs
        dif2_abs = dif2_0_abs + dif2_1_abs
        dif3_abs = dif3_0_abs + dif3_1_abs
    elif mode == 'dp':
        important_index1_Reg, important_index2_Reg, important_value1_Reg, important_value2_Reg = cal_importance_gapReg(
            model, optimizer_cal, X_test_0, X_test_1, 1, mode='dp')
        important_index1_CE, important_index2_CE, important_value1_CE, important_value2_CE = cal_importance(
            model, optimizer_cal, X_test, y_test, 1)
    else:
        print("Error: no mode in evaluate_difference()")

    index1_Reg_CE = [list(important_index1_CE).index(i) for i in important_index1_Reg]
    index2_Reg_CE = [list(important_index2_CE).index(i) for i in important_index2_Reg]

    important_index1 = important_index1_Reg
    important_index2 = important_index2_Reg
    print("*" * 20)
    print('important_index1', important_index1_Reg)
    print('important_index1_CE', important_index1_CE)
    print('important_index2', important_index2_Reg)
    print('important_index2_CE', important_index2_CE)
    print("*" * 20)
    for i in range(10):
        print(f"index1_Reg_CE_{10*i}%-{10*i + 10}%:", index1_Reg_CE[20*i:20*i+20])
        print(f"index2_Reg_CE_{10*i}%-{10*i + 10}%:", index2_Reg_CE[20*i:20*i+20])

    print('\ndiff_value1')
    print('top10%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[0:20]].mean().item())
    print('top10%-20%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[20:40]].mean().item())
    print('top20%-30%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[40:60]].mean().item())
    print('top30%-40%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[60:80]].mean().item())
    print('top40%-50%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[80:100]].mean().item())
    print('top50%-60%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[100:120]].mean().item())
    print('top60%-70%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[120:140]].mean().item())
    print('top70%-80%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[140:160]].mean().item())
    print('top80%-90%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[160:180]].mean().item())
    print('top90%-100%')
    print('important_avg_diff_value1: ', dif1_abs[important_index1[180:200]].mean().item())

    print('\ndiff_value2')
    print('top10%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[0:20]].mean().item())
    print('top10%-20%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[20:40]].mean().item())
    print('top20%-30%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[40:60]].mean().item())
    print('top30%-40%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[60:80]].mean().item())
    print('top40%-50%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[80:100]].mean().item())
    print('top50%-60%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[100:120]].mean().item())
    print('top60%-70%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[120:140]].mean().item())
    print('top70%-80%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[140:160]].mean().item())
    print('top80%-90%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[160:180]].mean().item())
    print('top90%-100%')
    print('important_avg_diff_value2: ', dif2_abs[important_index2[180:200]].mean().item())

    print('\ndiff_value3')
    print('important_avg_diff_value3: ', dif3_abs.mean().item())


def mask_neuron_test(model, X_test, y_test, A_test, criterion, optimizer=None, mode=None):
    model.eval()
    optimizer_cal = copy.deepcopy(optimizer)
    idx_00 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 0)[0]))
    idx_01 = list(set(np.where(A_test == 0)[0]) & set(np.where(y_test == 1)[0]))
    idx_10 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 0)[0]))
    idx_11 = list(set(np.where(A_test == 1)[0]) & set(np.where(y_test == 1)[0]))

    X_test_00 = X_test[idx_00]
    X_test_01 = X_test[idx_01]
    X_test_10 = X_test[idx_10]
    X_test_11 = X_test[idx_11]

    X_test_00 = torch.tensor(X_test_00).cuda().float()
    X_test_01 = torch.tensor(X_test_01).cuda().float()
    X_test_10 = torch.tensor(X_test_10).cuda().float()
    X_test_11 = torch.tensor(X_test_11).cuda().float()

    X_test = torch.tensor(X_test).cuda().float()
    y_test = torch.tensor(y_test).cuda().float()

    # diff
    X_test_0 = torch.cat((X_test_00, X_test_01), dim=0)
    X_test_1 = torch.cat((X_test_10, X_test_11), dim=0)

    torch.set_printoptions(precision=10, sci_mode=False)

    # importance

    # separate class
    batch_x_0_ = [X_test_00, X_test_01]
    batch_x_1_ = [X_test_10, X_test_11]

    if mode == 'eo':
        important_index1, important_index2, important_value1, important_value2 = cal_importance_gapReg(
            model, optimizer_cal, batch_x_0_, batch_x_1_, 1, mode='eo')
    elif mode == 'dp':
        important_index1, important_index2, important_value1, important_value2 = cal_importance_gapReg(
            model, optimizer_cal, X_test_0, X_test_1, 1, mode='dp')
    else:
        print("Error: no mode in evaluate_difference()")

    interval1 = len(important_index1) // 5
    interval2 = len(important_index2) // 5

    # no mask
    output = model.mask_forward(X_test)

    y_scores = output[:, 0].data.cpu().numpy()
    ap_no_mask = average_precision_score(y_test.cpu(), y_scores)
    loss_no_mask = criterion(output, y_test).item()
    print(f'no mask')
    print("loss:", loss_no_mask)
    print('ap:', ap_no_mask)

    # mask
    ap = []
    loss = []
    for j in range(5):
        output = model.mask_forward(X_test, important_index1[interval1 * j:interval1 * j + interval1],
                                         important_index2[interval2 * j:interval2 * j + interval2])

        y_scores = output[:, 0].data.cpu().numpy()
        ap_j = average_precision_score(y_test.cpu(), y_scores)
        ap.append(ap_j)

        loss_sup = criterion(output, y_test)
        loss.append(loss_sup.item())

    for i in range(5):
        print(f'{20 * i}%-{20 * i + 20}%:')
        print("loss:", loss[i])
        print('ap:', ap[i])
        print("loss_dif:", loss[i]-loss_no_mask)
        print('ap_dif:', ap[i]-ap_no_mask)

