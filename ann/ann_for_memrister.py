import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sb
import pandas as pd
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system('rm Accuracy.log')
# lambda1 = lambda epoch: epoch//30
# lambda2 = lambda epoch: 0.95**epoch

def write_accuracy_log(acc):
    sf = open('Accuracy.log','a+')
    sf.write(str(acc))
    sf.write('\n')
    sf.close()

class model(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(model, self).__init__()
        self.single_ann = torch.nn.Linear(n_feature, n_output)

    def forward(self, x):
        x = self.single_ann(x)
        out = F.log_softmax(x, dim=1)
        return out

batch_size = 256
num_epoch = 50
learning_rate = 1e-2

# device = torch.device('cuda:0')

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
# network.to(device)
# optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate) # weight_decay=1e-4)
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate) # weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda2)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.2)
criterion = torch.nn.CrossEntropyLoss()

num_data = len(train_dataset)
# training
loss_list, acc_list = [], []
batch_loss_list, batch_acc_list = [], []
test_acc_list, test_loss_list = [], []
for epoch in range(num_epoch):
    train_loss = 0
    acc = []
    for i_batch, (img, target) in enumerate(train_loader):
        # img, target = img.to(device), target.to(device)
        img = img.reshape(-1, 28*28)
        # optimizer.zero_grad()
        network.zero_grad()
        out = network(img)

        loss_batch = criterion(out, target)
        loss_batch.backward()

        optimizer.step()
        # scheduler.step()
        for p in network.parameters():
            if p.shape == (10,28*28):
                # weight_ori = p.data
                weight_ori = p.data * (torch.ones(p.data.shape) + 0.03*torch.randn(p.data.shape)) #0.0159
                p.data = weight_ori
        
        train_loss += loss_batch
        acc_batch = torch.sum(torch.argmax(out, dim=1) == target) / len(target)
        # write_accuracy_log(acc_batch)
        acc.append(acc_batch)
        if i_batch % 100 == 0:
            print('batch: %2d, loss: %f, acc: %f' % (i_batch, loss_batch, acc_batch))
        batch_loss_list.append(loss_batch)
        batch_acc_list.append(acc_batch)

    train_acc = (sum(acc) * batch_size / num_data).cpu().numpy()
    acc_list.append(train_acc)
    loss_list.append(train_loss)

    # test
    with torch.no_grad():
        val_loss, val_acc = 0, 0
        test_out, test_targets = [], []
        for i_batch, (img, target) in enumerate(test_loader):
            # img, target = img.to(device), target.to(device)
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
plt.plot(test_acc_list,'r',label='acc')
plt.legend(loc="upper right")
plt.xlabel('epoch')
plt.ylabel('acc')
plt.savefig('accuracy.png')
plt.clf()

    # # confusion matrix
    # conf_mat = confusion_matrix(test_targets.cpu().numpy(), torch.argmax(test_out, dim=1).cpu().numpy())
    # conf_mat_dataframe = pd.DataFrame(conf_mat, index=list(range(10)), columns=list(range(10)))

    # plt.figure(figsize=(12, 8))
    # sb.heatmap(conf_mat_dataframe, annot=True)
    # plt.savefig(f'conf_mat_0706_multi_pulse.png')


# save model
# torch.save(network.state_dict(), os.path.join(SAVE_PATH, saved_name))
# torch.save({'train acc': acc_list,
#             'train loss': loss_list,
#             'train batch acc': batch_acc_list,
#             'train batch loss': batch_loss_list,
#             'test acc': test_acc_list,
#             'test loss': test_loss_list
#             }, 'acc_loss_lists_0706_multi_pluse.pt')
