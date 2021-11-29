import json
import os
import numpy as np
import scipy.io as sio
import pandas as pd

sig_path = 'ECGdata/天池/data/train'  # 文件目录
label_path = 'ECGdata/天池/P_QRS_T_labels_1000/ManualAnnotation'  # 标签目录
key_path = 'tianchi'
key_sig = 'I II V1 V2 V3 V4 V5 V6'.split(' ')[0]
# 存储信号.npy文件
train_sigs = os.path.join(key_path, 'train_sigs')
train_labels = os.path.join(key_path, 'train_labels')
val_sigs = os.path.join(key_path, 'val_sigs')
val_labels = os.path.join(key_path, 'val_labels')

for i in [key_path, train_sigs, train_labels, val_sigs, val_labels]:
    if not os.path.exists(i):
        os.mkdir(i)

files = os.listdir(label_path)
files.sort()
# 划分
train_val = [files[:len(files) * 4 // 5], files[len(files) * 4 // 5:]]

from scipy import signal


def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def do_preproc(files, X_path, Y_path, key):
    for i in range(len(files)):
        file_name = files[i]
        print(i, '/', len(files), file_name)
        if not file_name.endswith('.json'):
            continue
        name = file_name[:-5]  # xx.json
        path_x = os.path.join(sig_path, name + '.txt')
        path_y = os.path.join(label_path, name + '.json')
        assert path_y == os.path.join(label_path, file_name)
        # read
        if not os.path.exists(path_x):
            path_x = os.path.join('ECGdata/天池/data/testA', name + '.txt')
        data = pd.read_csv(path_x, sep=' ')
        x = data[key].values
        x = x * 4.88
        assert len(x) == 5000
        x = resample(x, len(x)*360//500)  # 500Hz->360Hz

        with open(path_y, 'r') as load_f:
            data = json.load(load_f)  # ['P on', 'P off', 'R on', 'R off', 'T on', 'T off']
        # bg 0 p 1 qrs 2 t 3
        y = np.zeros(shape=x.shape)
        for i in range(3):
            t1 = ['P on', 'P off', 'R on', 'R off', 'T on', 'T off'][i * 2]
            t2 = ['P on', 'P off', 'R on', 'R off', 'T on', 'T off'][i * 2 + 1]
            t1 = data[t1]
            t2 = data[t2]
            assert len(t1) == len(t2)
            for j in range(len(t1)):
                a = round(t1[j] * 3600 / 5000)
                b = min(round(t2[j] * 3600 / 5000) + 1, 3600)
                y[a:b] = i + 1

        # write
        idx = 0
        while (idx + 1) * 1800 <= len(x):
            xi = x[idx * 1800:(idx + 1) * 1800]
            yi = y[idx * 1800:(idx + 1) * 1800]
            xi_path = os.path.join(X_path, "%s-%d.npy" % (name, idx))
            yi_path = os.path.join(Y_path, "%s-%d.npy" % (name, idx))
            np.save(xi_path, xi)
            np.save(yi_path, yi)
            idx += 1


do_preproc(train_val[0], train_sigs, train_labels, key_sig)
do_preproc(train_val[1], val_sigs, val_labels, key_sig)
