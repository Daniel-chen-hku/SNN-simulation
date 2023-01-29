#!/usr/bin/env python
import numpy as np
from import_set import load_ptb_xl_set,load_mit_ECG_set,load_sliced_ecg
from simulation_for_oect import snn_learn,snn_get
from data_class import ESN_ECG,RC_ECG,STDP_DATA
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = 'ptb' # mit
mode = 'snn'
#45 30train 15test 5test 
# predecoding 
class model(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(model, self).__init__()
        self.single_ann = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):
        x = self.single_ann(x)
        out = F.log_softmax(x, dim=1)
        return out

def ann_network(X_train,y_train,X_test,y_test):
    X_train,y_train = torch.Tensor(X_train),torch.Tensor(y_train)
    y_train = torch.argmax(y_train,dim=1)
    X_test,y_test = torch.Tensor(X_test),torch.Tensor(y_test)
    y_test = torch.argmax(y_test,dim=1)
    network = model(n_feature=12*13,n_output=5)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01) # weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    acc = []
    i_batch = 20
    for epoch in range(800):
        train_loss = 0
        test_out, test_targets = [], []
        # optimizer.zero_grad()
        network.zero_grad()
        out = network(X_train)
        loss_batch = criterion(out, y_train)
        loss_batch.backward()
        optimizer.step()
        train_loss += loss_batch
        acc_batch = torch.sum(torch.argmax(out, dim=1) == y_train) / len(y_train)
        acc.append(acc_batch)
        print('batch: %2d, loss: %f, acc: %f' % (i_batch, loss_batch, acc_batch))
        test_out.append(out)
        test_targets.append(y_train)
    plt.plot(acc)
    plt.savefig('acc.png')
    np.save('acc.npy',np.array(acc))
    test_targets = torch.cat(test_targets, dim=0)
    test_out = torch.cat(test_out, dim=0)
    # confusion matrix
    conf_mat = confusion_matrix(test_targets.cpu().numpy(), torch.argmax(test_out, dim=1).cpu().numpy())
    conf_mat_dataframe = pd.DataFrame(conf_mat, index=list(range(5)), columns=list(range(5)))
    plt.clf()
    plt.figure(figsize=(12, 8))
    sb.heatmap(conf_mat_dataframe, annot=True)
    plt.savefig(f'conf_mat_0706_multi_pulse.png')

def esn_for_ecg():
    # load ecg set
    if data == 'ptb':
        (X_train,train_label,X_test,test_label) = load_ptb_xl_set()
    elif data == 'mit':
        (X_train,train_label,X_test,test_label) = load_mit_ECG_set()
    esn = ESN_ECG(X_train,train_label,X_test,test_label,leaky=0.2)
    return esn.esn_evo()

def rc_for_ecg():
    """
    The ECG signal data should be sliced in sequence.
    If it is not sliced, the short-term memory characteristics of the device will cause the previous signal to be forgotten,
    and the final output of the one-dimensional signal will not have a high-precision classification effect. 
    """
    # norm:0 cd:1 hyp:2 mi:3 sttc:4
    (sample_cd,sample_hyp,sample_mi,sample_norm,sample_sttc) = load_sliced_ecg()
    # nine sample for each type, six for train ,three for test
    sample_norm = np.delete(sample_norm,6,axis = 0)
    sp_ay,tt_ay = np.array([0,1,2,4,6,8]),np.array([3,5,7])
    # X_train = np.array([sample_cd[sp_ay],sample_hyp[sp_ay],sample_mi[sp_ay],sample_norm[sp_ay],sample_sttc[sp_ay]])
    X_train = np.concatenate((sample_cd[sp_ay],sample_hyp[sp_ay],sample_mi[sp_ay],sample_norm[sp_ay],sample_sttc[sp_ay]))
    rs_ay,sg_ay = np.array([0,6,12,18,24]),np.array([1,1,1,1,1])
    # rs_ay = np.array([rs_ay,rs_ay+sg_ay,rs_ay+2*sg_ay,rs_ay+3*sg_ay,rs_ay+4*sg_ay,rs_ay+5*sg_ay])
    rs_ay = np.concatenate((rs_ay,rs_ay+sg_ay,rs_ay+2*sg_ay,rs_ay+3*sg_ay,rs_ay+4*sg_ay,rs_ay+5*sg_ay))
    X_train = X_train[rs_ay]
    y_train = np.array(6*[1,2,3,0,4])
    train_label = np.zeros((y_train.shape[0],y_train.max()+1))
    for i in range(y_train.shape[0]):
        train_label[i][y_train[i]] = 1
    # X_test = np.array([sample_cd[tt_ay],sample_hyp[tt_ay],sample_mi[tt_ay],sample_norm[tt_ay],sample_sttc[tt_ay]])
    X_test = np.concatenate((sample_cd[tt_ay],sample_hyp[tt_ay],sample_mi[tt_ay],sample_norm[tt_ay],sample_sttc[tt_ay]))
    y_test = np.array([1,1,1,2,2,2,3,3,3,0,0,0,4,4,4])
    test_label = np.zeros((y_test.shape[0],y_test.max()+1))
    for j in range(y_test.shape[0]):
        test_label[j][y_test[j]] = 1
    # (X_train,train_label,X_test,test_label)
    rc = RC_ECG(k=10,a=2)
    train = rc.rc_function(X_train - X_train.min())
    test = rc.rc_function(X_test - X_test.min())
    test = np.concatenate((train,test))
    test_label = np.concatenate((train_label,test_label))
    X_train = train
    X_test = test
    y_train = train_label
    y_test = test_label
    np.savez('rc_ecg.npz', train_data = X_train, train_label = train_label, test_data = X_test, test_label = test_label)
    if mode == 'ann':
        ann_network(train,train_label,test,test_label)
    elif mode == 'snn':
        stdp_data = STDP_DATA(learning_rate=1,g_initial=400,g_size=[train.shape[1],train_label.shape[1]],dynamic_rate = 'NO')
        snn_learn(train,train_label,test,test_label,stdp_data)
        snn_get(test,test_label,stdp_data)
        return (train,train_label,test,test_label)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists('rc_for_rcg'):
        os.system('mkdir rc_for_rcg')
    os.chdir('rc_for_rcg')
    rc_for_ecg()