import torch.optim as optim
import torchvision
import torch.utils.data as data

import matplotlib.pyplot as plt
import numpy as np
import argparse

from ProgressBar import ProgressBar
from model import *
from config import *

# record
best_acc = 0
count_early_stop = 0
record_train_loss = []
record_val_loss = []
record_train_acc = []
record_val_acc = []

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=configure['transform'])

train_set_size = int(len(trainset) * 0.8)
valid_set_size = len(trainset) - train_set_size
train_set, valid_set = data.random_split(trainset, [train_set_size, valid_set_size])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=configure['batch_size'], shuffle=True)
valloader = torch.utils.data.DataLoader(valid_set, batch_size=configure['batch_size'], shuffle=True)

print('===>Building Model')
net = JorNet().to(device)
try:
    from torchsummary import summary
    summary(net, (3, 32, 32))
except:
    print(net)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=configure['lr'], momentum=0.9)

def val():
    global best_acc
    global count_early_stop
    global record_val_loss
    correct = 0
    total = 0
    val_loss = 0.
    net.eval()
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            ProgressBar(batch_index, len(valloader), 'Loss:%.3f | Acc:%.3f (%d/%d)' % (val_loss/(batch_index+1), correct*100/total, correct, total))
    
    acc = correct * 100 / total
    if acc > best_acc:
        #TODO: save
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/model.pth')
        file  = open('./checkpoint/config.txt', 'w')
        for key in configure.keys():
            if key == 'transform':
                break
            file.write("{} : {}\n".format(key, configure[key]))
        file.close()

        best_acc = acc
        print("UPDATE:%.2f!!!!!!!!!!!!!!!!" % acc)
        count_early_stop = 0
    else:
        count_early_stop += 1

    record_val_acc.append(correct*100/total)
    record_val_loss.append(val_loss)
    print()

#train
def train(epoch):
    global record_train_loss
    print('EPOCH [%d/%d]' % (epoch+1, configure['num_epoch']))
    net.train()
    correct = 0
    total = 0
    train_loss = 0.
    for batch_index, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        
        ProgressBar(batch_index, len(trainloader), 'Loss:%.3f | Acc:%.3f (%d/%d)' % (train_loss/(batch_index+1), correct*100/total, correct, total))
    
    record_train_acc.append(correct*100/total)
    record_train_loss.append(train_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-lr", "--learning_rate", help="Learning Rate(default=0.001)", type=float, default=configure['lr'])
    parser.add_argument("-ne", "--num_epoch", help="Training times(default=200)", type=int, default=configure['num_epoch'])
    parser.add_argument("-bs", "--batch_size", help="The batch size the training data is sliced(default=8)", type=int, default=configure['batch_size'])
    parser.add_argument("-es", "--early_stop", help="Early stop if the accuracy didn't grow in setting steps(default=20)", type=int, default=configure['early_stop'])
    parser.add_argument("-s", "--show", action="store_true", help="show the training and accuracy timesteps graph after training finished")
    args = parser.parse_args()
    configure['lr'] = args.learning_rate
    configure['num_epoch'] = args.num_epoch
    configure['batch_size'] = args.batch_size
    configure['early_stop'] = args.early_stop
        
    for epoch in range(configure['num_epoch']):
        adjust_lr(optimizer, epoch)
        train(epoch)
        val()
        if count_early_stop >= configure['early_stop']:
            break

    print('===>Finish Training')

    if args.show:
        x = np.arange(1, len(record_train_loss) + 1)
        y1 = record_train_loss
        y2 = record_val_loss
        plt.plot(x,y1,label="train_loss")
        plt.plot(x,y2,label="test_loss")

        plt.title("LOSS",fontsize=15)
        plt.savefig('loss.png')
        plt.legend()
        plt.show()

        y1 = record_train_acc
        y2 = record_val_acc
        plt.plot(x,y1,label="train_acc")
        plt.plot(x,y2,label="test_acc")
        plt.title("Accuracy",fontsize=15)
        plt.savefig('acc.png')
        plt.legend()
        plt.show()

           