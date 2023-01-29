# from tensorflow.python.keras.layers.core import Dense
# from oect_plot import *
# from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt
# # from keras.utils import to_categorical
# from keras.datasets import mnist
# # from emnist import extract_training_samples,extract_test_samples
# # X_train, y_train = extract_training_samples('letters')
# # X_test, y_test = extract_test_samples('letters')
# # num_pixels = X_train.shape[1] * X_train.shape[2]
# # X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
# # X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# # X_train = X_train.astype('float32')/255
# # X_test = X_test.astype('float32')/255
# # y_train = to_categorical(y_train)
# # y_test = to_categorical(y_test)
# # (X_train, y_train), (X_test, y_test) = mnist.load_data()
# # num_pixels = X_train.shape[1] * X_train.shape[2]
# # X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
# # X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# # X_train = X_train.astype('float32')/255
# # X_test = X_test.astype('float32')/255
# # y_train = to_categorical(y_train)
# # y_test = to_categorical(y_test)

# model = keras.Sequential()
# model.add(Dense(units=10,activation='softmax', input_shape=(28*28,)))
# # model.add(Dense(units=27, activation='softmax'))
# print(model.summary())
# sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error',
#               optimizer=sgd,
#               metrics=['mae', 'acc'])

# model.fit(X_train, y_train, epochs=20)

# # test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
# loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

i_batch = 20

class model(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(model, self).__init__()
        self.single_ann = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):
        x = self.single_ann(x)
        out = F.log_softmax(x, dim=1)
        return out

dataset = np.load('rc_for_rcg/rc_ecg.npz')
X_train,y_train = dataset['train_data'],dataset['train_label']
X_test,y_test = dataset['test_data'],dataset['test_label']
X_train,y_train = torch.Tensor(X_train),torch.Tensor(y_train)
y_train = torch.argmax(y_train,dim=1)
X_test,y_test = torch.Tensor(X_test),torch.Tensor(y_test)
y_test = torch.argmax(y_test,dim=1)
network = model(n_feature=12*13,n_output=5)
optimizer = torch.optim.Adam(network.parameters(), lr=0.01) # weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# myNet = nn.Sequential(
#     nn.Linear(12*13,5),
#     F.log_softmax()
# )

# optimzer = torch.optim.SGD(myNet.parameters(), lr=0.05)
# loss_func = nn.MSELoss()
acc = []
weight = []
weight0 = np.zeros((5,12*13))
weight.append(weight0)
for epoch in range(800):
    train_loss = 0
    test_out, test_targets = [], []
    # optimizer.zero_grad()
    network.zero_grad()
    out = network(X_train)

    loss_batch = criterion(out, y_train)
    loss_batch.backward()

    optimizer.step()
    if (epoch+1) % 200 == 0:
        for p in network.parameters():
            if p.shape == (5,156):
                weight.append(p.data.numpy().astype('float32'))

    train_loss += loss_batch
    acc_batch = torch.sum(torch.argmax(out, dim=1) == y_train) / len(y_train)
    # write_accuracy_log(acc_batch)
    acc.append(acc_batch)
    # if i_batch % 100 == 0:
    print('batch: %2d, loss: %f, acc: %f' % (i_batch, loss_batch, acc_batch))
    test_out.append(out)
    test_targets.append(y_train)
    # print('epoch %3d, train loss: %f, train acc: %f, val loss: %f, val acc: %f'
    #       % (epoch, train_loss, train_acc, val_loss, val_acc))
    with torch.no_grad():
        val_loss, val_acc = 0, 0
        test_out, test_targets = [], []
        # img, target = img.to(device), target.to(device)
        img,target = X_test,y_test
        out = network(img)

        batch_loss = criterion(out, target)
        batch_acc = torch.sum(torch.argmax(out, dim=1) == target) / len(target)
        batch_acc = batch_acc.cpu().numpy()

        val_loss += batch_loss
        val_acc += batch_acc

        test_out.append(out)
        test_targets.append(target)

    test_targets = torch.cat(test_targets, dim=0)
    test_out = torch.cat(test_out, dim=0)

    val_acc = val_acc 
    print('epoch %3d, train loss: %f, val loss: %f, val acc: %f'
          % (epoch, train_loss, val_loss, val_acc))
# del(weight[-1])
vmin = np.min(weight)
vmax = np.max(weight)
np.save('ecg_ann_weight.npy',np.array(weight))
fig,ax = plt.subplots(1,len(weight))
for i in range(len(weight)):
    ax[i].matshow(weight[i], vmin=vmin, vmax=vmax, cmap='jet')
    ax[i].set_title('ep:'+str(i*5)+'k') # the first value is g initial
fig.tight_layout()
fig.savefig('weight_change.png')
plt.clf()
plt.cla()
plt.close()

# plt.plot(acc)
# plt.savefig('acc.png')
# np.save('acc.npy',np.array(acc))
# test_targets = torch.cat(test_targets, dim=0)
# test_out = torch.cat(test_out, dim=0)
# # confusion matrix
# conf_mat = confusion_matrix(test_targets.cpu().numpy(), torch.argmax(test_out, dim=1).cpu().numpy())
# # conf_mat = get_classification_metric(X_train.cpu().numpy(), y_train.cpu().numpy())
# conf_mat_dataframe = pd.DataFrame(conf_mat, index=list(range(5)), columns=list(range(5)))
# plt.figure(figsize=(12, 8))
# sb.heatmap(conf_mat_dataframe, annot=True)
# plt.savefig(f'conf_mat_0706_multi_pulse.png')
# for epoch in range(5000):
#     out = network(X_train)
#     loss = loss_func(out, y_train)  # 计算误差
#     optimzer.zero_grad()  # 清除梯度
#     loss.backward()
#     optimzer.step()

# print(myNet(X_train).data)