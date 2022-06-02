import torch
import torchvision

import matplotlib.pyplot as plt

from model import *
from config import *

PATH = './checkpoint/model.pth'

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=configure['transform'])
testloader = torch.utils.data.DataLoader(testset, batch_size=configure['batch_size'], shuffle=False)

def show_img():
    idxs = torch.randint(len(testset), (10, ))

    for i, idx in enumerate(idxs):
        plt.subplot(2, 5, i+1)
        img, label = testset[idx]
        img_cuda = img.to(device)
        outputs = net(img_cuda.view(1, 3, 32, 32))
        _, predict = torch.max(outputs, 1)
        plt.title(f'Predict:{predict.item()} {classes[predict.item()]}\nAnswer:{label} {classes[label]}', fontsize=10)
        img = (img + 1) / 2
        plt.imshow(img.permute(1, 2, 0))
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    plt.tight_layout()
    plt.show()


def test():
    record_correct = [0. for _ in range(len(classes))]
    record_total = [0. for _ in range(len(classes))]
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            for i in range(len(predicted)):
                if (predicted[i] == targets[i]):
                    record_correct[targets[i]] += 1
                record_total[targets[i]] += 1

    print(correct, type(correct))
    print(total, type(total))
    print('Accuracy: %.2f' % (correct * 100 / total))

    for i in range (len(classes)):
        print('Accuracy of %s : %2d %%' % (classes[i], record_correct[i] * 100 / record_total[i]))
    show_img()

if __name__ == '__main__':
    file  = open('./checkpoint/config.txt', 'r')
    for line in file:
        line = line.strip()
        key, data = line.split(' : ')
        configure[key] = float(data)
    file.close()
            
    net = JorNet().to(device)
    net.load_state_dict(torch.load(PATH))

    #test
    test()