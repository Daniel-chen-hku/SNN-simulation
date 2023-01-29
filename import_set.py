#!/usr/bin/env python
import numpy as np
import datetime
import sys
import os

def load_mnist() -> tuple:
    if os.path.exists('run_mnist_log/dataset.npz'):
        os.chdir('run_mnist_log/')
        dataset = np.load('mnist_dataset.npz')
        X_train,train_label = dataset['train_data'],dataset['train_label']
        X_test,test_label = dataset['test_data'],dataset['test_label']
        return (X_train,train_label,X_test,test_label)
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    train_label = np.zeros((y_train.shape[0],10))
    test_label = np.zeros((y_test.shape[0],10))
    for i in range(y_train.shape[0]):
        train_label[i][y_train[i]] = 1
    for j in range(y_test.shape[0]):
        test_label[j][y_test[j]] = 1
    if not os.path.exists('run_mnist_log'):
        cmd = 'mkdir' + ' ' + 'run_mnist_log/'
        os.system(cmd)
    dir = 'run_mnist_log/'
    os.chdir(dir)
    np.savez('mnist_dataset.npz', train_data = X_train, train_label = train_label, test_data = X_test, test_label = test_label)
    return (X_train,train_label,X_test,test_label)

def load_emnist() -> tuple:
    if os.path.exists('run_emnist_log/emnist_dataset.npz'):
        os.chdir('run_emnist_log/')
        dataset = np.load('emnist_dataset.npz')
        X_train,train_label = dataset['train_data'],dataset['train_label']
        X_test,test_label = dataset['test_data'],dataset['test_label']
        return (X_train,train_label,X_test,test_label)
    from emnist import extract_training_samples,extract_test_samples
    X_train, y_train = extract_training_samples('letters')
    X_test, y_test = extract_test_samples('letters')
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    vmax = np.max(y_train)
    vmin = np.min(y_train)
    train_label = np.zeros((y_train.shape[0],vmax-vmin+1))
    test_label = np.zeros((y_test.shape[0],vmax-vmin+1))
    for i in range(y_train.shape[0]):
        train_label[i][y_train[i]-vmin] = 1
    for j in range(y_test.shape[0]):
        test_label[j][y_test[j]-vmin] = 1
    if not os.path.exists('run_emnist_log'):
        cmd = 'mkdir' + ' ' + 'run_emnist_log/'
        os.system(cmd)
    dir = 'run_emnist_log/'
    os.chdir(dir)
    np.savez('emnist_dataset.npz', train_data = X_train, train_label = train_label, test_data = X_test, test_label = test_label)
    return (X_train,train_label,X_test,test_label)

def load_mit_ECG_set() -> tuple:
    import pickle
    from collections import Counter
    if os.path.exists('run_ECG_log/mit-bih-database.pkl'):
            # read pkl
        with open('run_ECG_log/mit-bih-database.pkl', 'rb') as fin:
            res = pickle.load(fin)
        ## scale data
        all_data = res['data']
        all_peak = res['peak']
        all_label = res['label']
        print(all_data.shape)
        label_type = []
        for i in range(0, all_peak.shape[0]):
            label_type = np.hstack([label_type, all_label[i]])
        print(Counter(label_type))
        os.chdir('run_ECG_log/')
        return
    import pandas as pd
    import wfdb
    from IPython.display import display
    from tqdm import tqdm
    data_path = '/home/chenxi/python_task/sim_for_oect/ECG/physionet.org/files/mitdb/1.0.0/'
    all_data = []
    all_peak = []
    all_label = []
    filenames = pd.read_csv(os.path.join(data_path, 'RECORDS'), header=None)
    filenames = filenames.iloc[:, 0].values  # 拿第一列的值
    # print(filenames)  # 打印文件名
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
        all_label.append(np.array(atr.symbol))
    all_data = np.array(all_data)  # list转nparray
    all_peak = np.array(all_peak)
    all_label = np.array(all_label)
    if not os.path.exists('run_ECG_log'):
        cmd = 'mkdir' + ' ' + 'run_ECG_log/'
        os.system(cmd)
    dir = 'run_ECG_log/'
    os.chdir(dir)
    res = {'data': all_data, 'peak': all_peak, 'label': all_label}  # res 作为json格式保存数据和标签
    # display(res)
    with open('run_ECG_log/mit-bih-database.pkl', 'wb') as fout:
        pickle.dump(res, fout)
    return

# def deal_ecg_packet(X_train,train_label,X_test,test_label):

def load_ptb_xl_set() -> tuple:
    if os.path.exists('ECG/ptb_dataset.npz'):
        dataset = np.load('ECG/ptb_dataset.npz',allow_pickle=True)
        X_train,train_label = dataset['train_data'],dataset['train_label']
        X_test,test_label = dataset['test_data'],dataset['test_label']
        if os.path.exists('run_ecg_log'):
            os.chdir('run_ecg_log/')
            return (X_train,train_label,X_test,test_label)
        cmd = 'mkdir' + ' ' + 'run_ecg_log/'
        os.system(cmd)
        os.chdir('run_ecg_log/')
    sys.path.append('/home/chenxi/python_task/sim_for_oect/ECG/')
    from test_for_PTB_XL import create_npz_set
    (X_train,train_label,X_test,test_label) = create_npz_set()
    return (X_train,train_label,X_test,test_label)

def load_sliced_ecg() -> tuple:
    path = '/home/hegan/ECG/Disease/'
    # dataset = np.load(path+'sliced_ecg.npz')
    # X_train,train_label = dataset['train_data'],dataset['train_label']
    # X_test,test_label = dataset['test_data'],dataset['test_label']
    # return (X_train,train_label,X_test,test_label)
    # 10sample*12channel*13*90
    # 10*12*13
    cd_set = np.load(path+'sample_CD10.npz')
    sample_cd = cd_set['train_data']
    hyp_set = np.load(path+'sample_HYP10.npz')
    sample_hyp = hyp_set['train_data']
    mi_set = np.load(path+'sample_MI10.npz')
    sample_mi = mi_set['train_data']
    norm_set = np.load(path+'sample_NORM10.npz')
    sample_norm = norm_set['train_data']
    sttc_set = np.load(path+'sample_STTC10.npz')
    sample_sttc = sttc_set['train_data']
    return (sample_cd,sample_hyp,sample_mi,sample_norm,sample_sttc)

def set_working_path():
    # os.chdir(os.path.dirname(__file__))
    # time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    time_str = 'oect_snn_lr_1_epoch_20'
    cmd = 'mkdir' + ' ' + time_str + 'log/'
    os.system(cmd)
    cmd = time_str + 'log/'
    os.system('cp *dataset.npz'+ ' ' + cmd)
    os.chdir(cmd)

def write_system_log(trainingset,teacherset,testset,answer):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'snn' + '_simulation' + time_str + 'log.ini'
    sf = open(filename,'w+')
    sf.write('learning sample:\n')
    for i in range(trainingset.shape[0]):
        sf.write(str(trainingset[i]))
        sf.write('\n')
        sf.write(str(teacherset[i]))
        sf.write('\n')
    sf.write('testing sample:\n')
    for i in range(testset.shape[0]):
        sf.write(str(testset[i]))
        sf.write('\n')
        sf.write(str(answer[i]))
        sf.write('\n')
    sf.close()
