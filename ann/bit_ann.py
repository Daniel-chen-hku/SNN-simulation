import torch
import torch.nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import pandas as pd
import os
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
mode = '9bits' if len(sys.argv) <= 1 else str(sys.argv[1])
learning_rate = 0.01 if len(sys.argv) <= 2 else float(sys.argv[2])

filename = 'ann_5bits_'+str(learning_rate).replace('.','') if mode == '5bits' else 'ann_9bits_'+str(learning_rate).replace('.','')

def write_accuracy_log(acc):
    sf = open('ann_bits/dobule_layer/acc_'+filename+'.log','a+')
    sf.write(str(acc))
    sf.write('\n')
    sf.close()

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

file_5bits = 'lenet_5/OECTS_5bits.xlsx'
file_9bits = 'lenet_5/OECTS_9bits.xlsx'
data_5bits = pd.read_excel(file_5bits)
data_9bits = pd.read_excel(file_9bits)
data_5bits = np.array(data_5bits)
data_9bits = np.array(data_9bits)
bits5 = data_5bits[1:52,9].astype('float32')
bits9 = data_9bits[0:1024,1].astype('float32')
bits5_array = normalization(bits5)
bits9_array = normalization(bits9)
bits5_array = torch.from_numpy(bits5_array)
bits9_array = torch.from_numpy(bits9_array)
normalized_array = bits5_array if mode == '5bits' else bits9_array

def find_nearest(value):
    # array = np.asarray(array)
    # array = bits5_array
    idx = (torch.abs(normalized_array - value)).argmin()
    return normalized_array[idx]

batch_size = 256
epoch = 6

class model(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(model, self).__init__()
        self.annl1 = torch.nn.Linear(n_feature, 50)
        self.annl2 = torch.nn.Linear(50,n_output)

    def forward(self, x):
        # x = F.relu(self.annl1(x))
        x = torch.sigmoid(self.annl1(x))
        x = self.annl2(x)
        out = F.log_softmax(x, dim=1)
        return out

batch_size = 128
num_epoch = 20


train_dataset = datasets.MNIST(
                               'data/',
                               train=True,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                         transforms.ToTensor()]))
num_data = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_dataset = datasets.MNIST(
    'data/', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
network = model(n_feature=28*28,n_output=10)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate) # weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

num_data = len(train_dataset)
# training
loss_list, acc_list = [], []
batch_loss_list, batch_acc_list = [], []
# test
test_acc_list, test_loss_list = [], []
for epoch in range(num_epoch):
    train_loss = 0
    acc = []
    for i_batch, (img, target) in enumerate(train_loader):
        img = img.reshape(-1, 28*28)
        # optimizer.zero_grad()
        network.zero_grad()
        out = network(img)
        loss_batch = criterion(out, target)
        loss_batch.backward()
        optimizer.step()
        for p in network.parameters():
            # if p.shape == (10,28*28):
            weight_ori = p.data * (torch.ones(p.data.shape) + 0.015*torch.randn(p.data.shape)) #0.0159
            # p.data = weight_ori
            max_weight = weight_ori.max()
            p.data = (weight_ori/max_weight).apply_(find_nearest) * max_weight

        train_loss += loss_batch
        acc_batch = torch.sum(torch.argmax(out, dim=1) == target) / len(target)
        acc.append(acc_batch)
        if i_batch % 10 == 0:
            print('batch: %2d, loss: %f, acc: %f' % (i_batch, loss_batch, acc_batch))
        batch_loss_list.append(loss_batch)
        batch_acc_list.append(acc_batch)

    train_acc = (sum(acc) * batch_size / num_data).cpu().numpy()
    acc_list.append(train_acc)
    loss_list.append(train_loss)

    with torch.no_grad():
        val_loss, val_acc = 0, 0
        test_out, test_targets = [], []
        for i_batch, (img, target) in enumerate(test_loader):
            img = img.reshape(-1, 28*28)
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

    val_acc = val_acc * batch_size / len(test_dataset)
    test_acc_list.append(val_acc)
    test_loss_list.append(val_loss)
    print('epoch %3d, train loss: %f, train acc: %f, val loss: %f, val acc: %f'
          % (epoch, train_loss, train_acc, val_loss, val_acc))
    write_accuracy_log(val_acc)
plt.clf()
plt.plot(test_acc_list,'r',label='error rate')
# plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('acc')
plt.savefig('ann_bits/dobule_layer/'+filename+'.png')
plt.clf()