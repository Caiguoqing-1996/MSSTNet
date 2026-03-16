#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Usage:
    BNCI2014_001:

'''

from DataPrepare import PrepareData
from ModelPrepare import model_prepare
from prepocess_in_model import filter_EEG
from train_model_withMean_twoStage import train_in_one_fold
import numpy as np
import torch
import torch.nn as nn

CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def CrossValidation_BCI4_2a(data_sel, model_sel, model_para, train_para):
    # 如果传入的是 None，初始化为空字典，以便后续写入电极位置等基础参数
    if model_para is None:
        model_para = {}

    data_prepare = PrepareData(data_sel=data_sel, win_classify=train_para['win_classify'])
    alldata = {'X': [],
               'Y': []}
    all_acc_test, all_model, all_precision, all_recall, all_f1, all_kappa = [], [], [], [], [], []

    for subi in data_prepare.sub_name[:]:
        for emptyi in range(100):
            torch.cuda.empty_cache()

        train_test = data_prepare.load_data_sub(target_sub=subi)

        train_data = filter_EEG(model_sel, data_sel, train_test['train'], train_test['fs'],
                                train_test['win_sel'], train_para['win_classify'])
        test_data = filter_EEG(model_sel, data_sel, train_test['test'], train_test['fs'],
                               train_test['win_sel'], train_para['win_classify'])

        fs = train_data['fs']
        n_repeat = train_para['n_repeat']
        acc_test_repeati = []
        precision_repeati = []
        recall_repeati = []
        f1_repeati = []
        kappa_repeati = []

        for repeati in range(n_repeat):
            print('*' * 100)
            print('Sub: ' + str(subi))
            print('model: ' + model_sel)
            print('repeati: ' + str(repeati))
            print('*' * 100)

            model_para['channel'] = 22
            model_para['time'] = (int(train_para['win_classify'][1] * fs) - int(train_para['win_classify'][0] * fs))
            model_para['class'] = 4
            model_para['fs'] = fs

            model_para['ch_names'] = [
                'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2', 'POz'
            ]
            model_para['pos'] = [
                [-0.00122927, 0.09327445, 0.10263929],  # Fz
                [-0.06185234, 0.05713329, 0.09376583],  # FC3
                [-0.03571586, 0.06171406, 0.11798302],  # FC1
                [-0.00127143, 0.06345979, 0.12657198],  # FCz
                [0.03313098, 0.06182849, 0.1167817],  # FC2
                [0.06062548, 0.05770742, 0.09379462],  # FC4
                [-0.0820853, 0.01929363, 0.06948967],  # C5
                [-0.06714872, 0.02335823, 0.10451068],  # C3
                [-0.03793782, 0.02633745, 0.12977061],  # C1
                [-0.00137413, 0.02761709, 0.14019949],  # Cz
                [0.03589272, 0.02635814, 0.12841234],  # C2
                [0.06532888, 0.0235731, 0.10369243],  # C4
                [0.08165317, 0.01969566, 0.06948181],  # C6
                [-0.06547223, -0.0118966, 0.10777792],  # CP3
                [-0.03742513, -0.01082424, 0.13344371],  # CP1
                [-0.0015249, -0.01051838, 0.14154914],  # CPz
                [0.03647196, -0.01090379, 0.13281227],  # CP2
                [0.06469626, -0.01199118, 0.10771287],  # CP4
                [-0.03065362, -0.04492739, 0.11947205],  # P1
                [-0.00170945, -0.04521299, 0.12667292],  # Pz
                [0.02988639, -0.04503254, 0.12074782],  # P2
                [-0.00189982, -0.0680541, 0.09591],  # POz
            ]

            model, loss_func = model_prepare(model_sel, model_para)

            acc_test, last_model, precision, f1, kappa \
                = train_in_one_fold(train_set_all=train_data, test_set=test_data,
                                    model=model, loss_func=loss_func, train_para=train_para)

            acc_test_repeati.append(acc_test)
            precision_repeati.append(precision)
            f1_repeati.append(f1)
            kappa_repeati.append(kappa)
            all_model.append(last_model)

        all_acc_test.append(np.mean(np.array(acc_test_repeati)))
        all_precision.append(np.mean(np.array(precision_repeati)))
        all_f1.append(np.mean(np.array(f1_repeati)))
        all_kappa.append(np.mean(np.array(kappa_repeati)))

    return all_acc_test, all_model, all_precision, all_f1, all_kappa


if __name__ == "__main__":
    '''
        'FBCNet', 'EEGNet', 'DeepNet', 'EEGConformer', 'ShallowNet', 'LDMANet'
        'IFNet', 'TSception', 'SMT_2a',  'TransNet',  'EDPNet'
    '''

    for i in range(100):
        torch.cuda.empty_cache()
    print("CUDA Available:", torch.cuda.is_available())

    data_sel = 'BNCI2014_001'
    model_sel = 'MSSTNet'

    # 将原来分散的参数统一写到 train_para 中
    train_para = {
        'win_classify': [0, 4],
        'lr': 0.001,
        'batch_size': 64,
        'n_repeat': 1,  # 这里如果只是测试跑一次流程，建议改小，原代码是 10000 可能会跑很久
        'train_prop': 0.8,
        'first_epochs': 1000,
        'second_epoch': 200,
        'min_train_epoch': 100,
        'min_second_epoch': 0,
        'patience': 100,
        'stop_criteria': 'accuracy',
    }

    # 1. 运行验证（模型参数直接给 None）
    all_acc_test, _, all_precision, all_f1, all_kappa \
        = CrossValidation_BCI4_2a(data_sel, model_sel, model_para=None, train_para=train_para)

    # 2. 按照个体顺序进行终端打印
    print("\n" + "=" * 60)
    print(f"模型: {model_sel} | 数据集: {data_sel} | 各个体测试结果")
    print("=" * 60)

    for idx in range(len(all_acc_test)):
        print(f"Subject {idx + 1:02d} => "
              f"Acc: {all_acc_test[idx]:.4f} | "
              f"Kappa: {all_kappa[idx]:.4f} | "
              f"F1: {all_f1[idx]:.4f} | "
              f"Precision: {all_precision[idx]:.4f}")

    # 3. 计算并打印整体均值和方差
    mean_acc, std_acc = np.mean(all_acc_test), np.std(all_acc_test)
    mean_kappa, std_kappa = np.mean(all_kappa), np.std(all_kappa)
    mean_f1, std_f1 = np.mean(all_f1), np.std(all_f1)

    print("-" * 60)
    print(f"总体平均 Acc   : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"总体平均 Kappa : {mean_kappa:.4f} ± {std_kappa:.4f}")
    print(f"总体平均 F1    : {mean_f1:.4f} ± {std_f1:.4f}")
    print("=" * 60)
    print("测试运行结束！")