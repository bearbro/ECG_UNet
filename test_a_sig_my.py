# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:08:55 2019

@author: Winham

test_a_sig.py: 加载训练好的模型，从验证集中随机选取一条信号进行测试

"""

import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import optimizers
from sklearn import preprocessing as prep
import matplotlib.pyplot as plt
import time

from Unet import Unet, Unet_crf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def lr_schedule(epoch):
    # 训练网络时学习率衰减方案
    lr = 0.0001
    if epoch >= 50:
        lr = 0.00001
    print('Learning rate: ', lr)
    return lr


dateSet = ["tianchi", "ccdd"][1]
train_batch_size = 4
n_classes = 4
input_length = 1800
optimizer_name = optimizers.Adam(lr_schedule(0))
model_name = ['base', "add_crf"][1]
tag = str(train_batch_size) + model_name

val_sig_path = '%s/val_sigs/' % (dateSet)
val_label_path = '%s/val_labels/' % (dateSet)

sig_files = os.listdir(val_sig_path)
label_files = os.listdir(val_label_path)

select = np.random.choice(sig_files, 1)[0]
# select="819-0.npy"
a_sig = np.load(val_sig_path + select)
a_seg = np.load(val_label_path + select)

K.clear_session()
tf.reset_default_graph()
if model_name == 'base':
    model = Unet(n_classes, input_length=input_length)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer_name,
                  metrics=['accuracy'])

elif model_name == 'add_crf':
    model = Unet_crf(n_classes, input_length=input_length, optimizer=optimizer_name)

model_path = 'myNet-%s-%s.h5' % (dateSet, tag)
model.load_weights(model_path)

a_sig = np.expand_dims(prep.scale(a_sig), axis=1)
a_sig = np.expand_dims(a_sig, axis=0)

tic = time.time()
a_pred = model.predict(a_sig)
toc = time.time()

print('Elapsed time: ' + str(toc - tic) + ' seconds.')

a_true = np.zeros(shape=a_pred.shape)
for idx, v in enumerate(a_seg):
    a_true[0, idx, int(v)] = 1


def plot(a_sig, a_pred, xy,title):
    plt.subplot(xy)
    plt.plot(a_sig[0, :, 0])
    plt.grid(True)
    plt.title(title)
    plt.plot(a_pred[0, :, 0], 'k')
    plt.plot(a_pred[0, :, 1], 'b')
    plt.plot(a_pred[0, :, 2], 'r')
    plt.plot(a_pred[0, :, 3], 'g')
    plt.legend(['Sig', 'Background', 'P', 'R', 'T'], loc='lower right')

title='%s-%s-%s' % (select[:-4],dateSet, tag)
plt.figure(figsize=(28, 12))

plot(a_sig, a_pred, 211,title=title+'-pred')

plot(a_sig, a_true, 212,title=title+'true')

dir_path="examples"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
plt.savefig(os.path.join(dir_path,"%s.png"%title))
# plt.show()
