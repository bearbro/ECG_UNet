# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:52:07 2019

@author: Administrator

train.py: 训练模型

"""
import os

import numpy as np

from Unet import Unet,Unet_crf
import LoadBatches1D_my
import tensorflow as tf
import keras
from keras import optimizers
from keras.models import load_model
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras_contrib.layers import CRF

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
warnings.filterwarnings("ignore")


def lr_schedule(epoch):
    # 训练网络时学习率衰减方案
    lr = 0.0001
    if epoch >= 50:
        lr = 0.00001
    print('Learning rate: ', lr)
    return lr


dateSet = ["tianchi","ccdd"][1]

train_sigs_path = '%s/train_sigs/' % (dateSet)
train_segs_path = '%s/train_labels/' % (dateSet)
train_batch_size = 4
n_classes = 4  # bg p qrs t
class_value = [0, 1, 2, 3]
input_length = 1800  # 5秒
optimizer_name = optimizers.Adam(lr_schedule(0))
val_sigs_path = '%s/val_sigs/' % (dateSet)
val_segs_path = '%s/val_labels/' % (dateSet)
val_batch_size = train_batch_size
model_name=['base',"add_crf"][1]
tag = str(train_batch_size)+model_name

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

if model_name=='base':
    model = Unet(n_classes, input_length=input_length)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer_name,
                  metrics=['accuracy'])

elif model_name=='add_crf':
    model = Unet_crf(n_classes, input_length=input_length,optimizer=optimizer_name)

model.summary()

output_length = 1800

train_size = len(os.listdir(train_sigs_path))
val_size = len(os.listdir(val_sigs_path))

print("train size", train_size, "val size", val_size)

G = LoadBatches1D_my.SigSegmentationGenerator(train_sigs_path, train_segs_path,
                                              train_batch_size, n_classes,
                                              output_length, class_value=class_value)

G2 = LoadBatches1D_my.SigSegmentationGenerator(val_sigs_path, val_segs_path,
                                               val_batch_size, n_classes, output_length,
                                               class_value=class_value)
model_path = 'myNet-%s-%s.h5' % (dateSet, tag)
if not os.path.exists(model_path):
    if model_name == 'base':
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                       monitor='val_acc', mode='max', save_best_only=True)
    elif model_name == 'add_crf':
        checkpointer = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                       monitor='val_crf_viterbi_accuracy', mode='max', save_best_only=True)

    # todo 乱序
    history = model.fit_generator(G, train_size // train_batch_size,
                                  validation_data=G2, validation_steps=val_size // val_batch_size, epochs=70,
                                  callbacks=[checkpointer, lr_scheduler])

    plt.figure()
    if model_name=='add_crf':
        plt.plot(history.history['crf_viterbi_accuracy'])
        plt.plot(history.history['val_crf_viterbi_accuracy'])
    else:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.savefig('acc-%s-%s.png' % (dateSet, tag))
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.savefig('loss-%s-%s.png' % (dateSet, tag))
    plt.show()

model.load_weights(model_path)
G2 = [G2.__next__() for i in range(val_size // val_batch_size)]
# ac_dict = [0] * 4
# sum_dict = [0] * 4
#
# for xy in tqdm(G2):
#     x, y = xy
#     y_pred = model.predict(x)
#     y_pred = np.argmax(y_pred, axis=-1)
#     y = np.argmax(y, axis=-1)
#     for ii in range(len(y)):
#         for i in range(len(y[ii])):
#             ac_dict[y[ii][i]] += 1 if y[ii][i] == y_pred[ii][i] else 0
#             sum_dict[y[ii][i]] += 1
#
# acc = np.sum(ac_dict) / np.sum(sum_dict)
# print("All-acc", acc)
# keys = ['bg', 'p', 'r', 't']
# for i in range(len(ac_dict)):
#     acci = np.sum(ac_dict[i]) / np.sum(sum_dict[i])
#     print("%s-acc" % keys[i], acci)

p_dict = [0] * 4
r_dict = [0] * 4
p_sum_dict = [0] * 4
r_sum_dict = [0] * 4
for xy in tqdm(G2):
    x, y = xy
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=-1)
    y = np.argmax(y, axis=-1)
    for i in range(len(r_dict)):
        r_dict[i] += np.sum((y == y_pred) & (y == i))
        p_dict[i] += np.sum((y == y_pred) & (y_pred == i))
        r_sum_dict[i] += np.sum(y == i)
        p_sum_dict[i] += np.sum(y_pred == i)

acc = np.sum(r_dict) / np.sum(r_sum_dict)
print("All-acc", acc)
keys = ['bg', 'p', 'r', 't']
for i in range(len(r_dict)):
    pi = np.sum(p_dict[i]) / np.sum(p_sum_dict[i])
    print("%s-p" % keys[i], np.sum(p_dict[i]), '/', np.sum(p_sum_dict[i]), "=", pi)
    ri = np.sum(r_dict[i]) / np.sum(r_sum_dict[i])
    print("%s-r" % keys[i], np.sum(r_dict[i]), '/', np.sum(r_sum_dict[i]), "=", ri)
