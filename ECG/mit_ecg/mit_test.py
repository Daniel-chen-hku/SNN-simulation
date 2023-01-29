### Import required packages
from IPython.display import display

import numpy as np
import pandas as pd
import pickle as dill
import os
from tqdm import tqdm
from collections import OrderedDict, Counter

import wfdb

# 读取数据、波峰、标签保存到mit-bih-database.pkl
def read_dataset(data_path):
    """
    before running this data preparing code,
    please first download the raw data from https://www.physionet.org/content/mitdb/1.0.0/,
    and put it in data_path
    """

    #  read data and label
    all_data = []
    all_peak = []
    all_label = []
    filenames = pd.read_csv(os.path.join(data_path, 'RECORDS'), header=None)
    filenames = filenames.iloc[:, 0].values  # 拿第一列的值
    print(filenames)  # 打印文件名
    for filename in tqdm(filenames):
        # read data
        dat = wfdb.rdrecord(os.path.join(data_path, '{0}'.format(filename)), channels=[0])
        dat = np.array(dat.p_signal)
        x = []
        for i in dat:
            x.append(i[0])
        all_data.append(x)
        # read label
        atr = wfdb.rdann(os.path.join(data_path, '{0}'.format(filename)), 'atr')
        all_peak.append(atr.sample)  # 无法转矩阵，因为每个人波峰个数不一样
        label = np.array(atr.symbol)
        all_label.append(np.array(atr.symbol))
    all_data = np.array(all_data)  # list转nparray
    all_peak = np.array(all_peak)
    all_label = np.array(all_label)
    res = {'data': all_data, 'peak': all_peak, 'label': all_label}  # res 作为json格式保存数据和标签
    display(res)
    with open(os.path.join(data_path, 'mit-bih-database.pkl'), 'wb') as fout:
        dill.dump(res, fout)


def preprocess_dataset(data_path):
    # read pkl
    with open(os.path.join(data_path, 'mit-bih-database.pkl'), 'rb') as fin:
        res = dill.load(fin)

    ## scale data
    all_data = res['data']
    all_peak = res['peak']
    all_label = res['label']

    print(all_data.shape)

    label_type = []
    for i in range(0, all_peak.shape[0]):
        label_type = np.hstack([label_type, all_label[i]])
    print(Counter(label_type))


data_path = './physionet.org/files/mitdb/1.0.0'
read_dataset(data_path)
preprocess_dataset(data_path)
