import os
import xml.dom.minidom
from scipy import signal
import numpy as np

path = 'ECGdata/CCDD/data1-943/data'  # 原始文件目录

key_path = 'ccdd'
key_sig = \
    ['MDC_ECG_LEAD_I', 'MDC_ECG_LEAD_II', 'MDC_ECG_LEAD_III', 'MDC_ECG_LEAD_V1', 'MDC_ECG_LEAD_V2', 'MDC_ECG_LEAD_V3',
     'MDC_ECG_LEAD_V4', 'MDC_ECG_LEAD_V5', 'MDC_ECG_LEAD_V6', 'MDC_ECG_LEAD_aVF', 'MDC_ECG_LEAD_aVL',
     'MDC_ECG_LEAD_aVR'][0]


def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


# 存储信号.npy文件
train_sigs = os.path.join(key_path, 'train_sigs')
train_labels = os.path.join(key_path, 'train_labels')
val_sigs = os.path.join(key_path, 'val_sigs')
val_labels = os.path.join(key_path, 'val_labels')

for i in [key_path, train_sigs, train_labels, val_sigs, val_labels]:
    if not os.path.exists(i):
        os.mkdir(i)

files = os.listdir(path)
files.sort()
# 划分
train_val = [files[:len(files) * 4 // 5], files[len(files) * 4 // 5:]]


def read_sig_label(xml_files, sig_key):
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(xml_files)
    collection = DOMTree.documentElement

    # 获取所有导联
    digits = collection.getElementsByTagName("digits")
    # 获取导联对应导名字
    digits_value = [di.childNodes[0].data for di in digits]
    # digits_name = [di.parentNode.previousSibling.previousSibling.getAttribute("code") for di in digits]
    digits_name = [di.parentNode.parentNode.getElementsByTagName("code")[0].getAttribute("code") for di in digits]
    assert len(digits_value) == len(digits_name)
    sig = {digits_name[i]: digits_value[i] for i in range(len(digits_name))}[sig_key]

    sig = np.array(list(map(int, sig.strip().split(" "))))
    sig = sig * 4.76837

    x = resample(sig, len(sig) * 360 // 500)  # 500Hz->360Hz

    # 获取标签
    # 获取所有时间区间
    # <value xsi:type="ILV_PQ">
    low_high = [i for i in collection.getElementsByTagName("value") if
                i.hasAttribute("xsi:type") and i.getAttribute("xsi:type") == 'ILV_PQ']
    # low_high_type=[i.parentNode.parentNode.parentNode.parentNode.previousSibling.previousSibling.getAttribute("code") for i in low_high]
    low_high_type = [
        i.parentNode.parentNode.parentNode.parentNode.parentNode.getElementsByTagName("value")[0].getAttribute("code")
        for i in low_high]
    assert len(low_high_type) == len(low_high)
    label = {'MDC_ECG_WAVC_PWAVE': 1, 'MDC_ECG_WAVC_QRSWAVE': 2, 'MDC_ECG_WAVC_TWAVE': 3}
    low_high_set = {'MDC_ECG_WAVC_PWAVE': set(), 'MDC_ECG_WAVC_QRSWAVE': set(), 'MDC_ECG_WAVC_TWAVE': set()}
    for i in range(len(low_high)):
        a = low_high[i].getElementsByTagName("low")
        b = low_high[i].getElementsByTagName("high")
        a = int(a[0].getAttribute("value")) if len(a) > 0 else None
        b = int(b[0].getAttribute("value")) if len(b) > 0 else None
        if a is not None and b is not None:
            low_high_set[low_high_type[i]].add((a, b))

    y = np.zeros(shape=x.shape)
    for k, v in low_high_set.items():
        for ab in v:
            ki = label[k]
            a, b = ab
            # 时间 -> idx
            a = round(a * 360 / 1000)
            b = len(x) if b is None else round(b * 360 / 1000) + 1
            b = min(len(x), b)
            y[a:b] = ki

    return x, y


def do_preproc(files, X_path, Y_path, sig_key):
    for i in range(len(files)):
        file_name = files[i]
        print(i, '/', len(files), file_name)
        if not file_name.endswith('.xml'):
            continue
        name = file_name[:-4]  # xx.xml
        path_xml = os.path.join(path, name + '.xml')
        # read
        x, y = read_sig_label(path_xml, sig_key)

        # write
        idx = 0
        while (idx + 1) * 1800 <= len(x):# 最后一段长度小于1800的直接舍弃
            xi = x[idx * 1800:(idx + 1) * 1800]
            yi = y[idx * 1800:(idx + 1) * 1800]
            xi_path = os.path.join(X_path, "%s-%d.npy" % (name, idx))
            yi_path = os.path.join(Y_path, "%s-%d.npy" % (name, idx))
            np.save(xi_path, xi)
            np.save(yi_path, yi)
            idx += 1

x,y=read_sig_label('ECGdata/CCDD/data1-943/data541-943/819.xml','MDC_ECG_LEAD_I')

do_preproc(train_val[0], train_sigs, train_labels, key_sig)
do_preproc(train_val[1], val_sigs, val_labels, key_sig)
