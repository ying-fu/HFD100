from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import torch
import sklearn
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from resnet import ResNet18
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import auxil

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
EPOCH = 100   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 32      #批处理尺寸(batch_size)
LR = 0.001        #学习率


# 定义train/validation数据集加载器

class Dataset_HSI(Dataset):
    def __init__(self, hsi_dataset, label, size=None, upsample=False):
        super(Dataset, self).__init__()
        self.hsi_dataset = hsi_dataset
        self.label = label
        self.size = size
        self.upsample = upsample

    def __len__(self):
        return self.size or len(self.hsi_dataset)

    def __getitem__(self, idx):
        hsi = self.hsi_dataset[idx]
        label = self.label[idx]

        return hsi, label


def load_split_train_test():

    train = h5py.File('data/scenes100/train/train3.h5', 'r')
    train.keys()
    hsi = train['HSI'][:]
    label = train['label'][:]
    train_data = Dataset_HSI(hsi, label)
    test = h5py.File('data/scenes100/test/test3.h5', 'r')
    test.keys()
    test_hsi = test['HSI'][:]
    test_label = test['label'][:]
    test_data = Dataset_HSI(test_hsi, test_label)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    return train_loader,test_loader

trainloader,testloader = load_split_train_test()
# 模型定义-ResNet
net = ResNet18(3,100).to(device)
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
#optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
print(net)
# 训练
if __name__ == "__main__":
    best_acc = 50  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("results/acc_3resnet_S100_HSI.txt", "w") as f:
        with open("results/log3S100re.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                exp_lr_scheduler.step()
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # print(len(inputs))
                    # print(length)
                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    # print(loss.item())
                    # print(sum_loss)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.long().data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    # print(correct)
                    # print(total)
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")

                y_pred = []
                y_labels = []
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        [y_pred.append(a) for a in outputs.data.cpu().numpy()]
                        [y_labels.append(a) for a in labels.long().cpu().numpy()]
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.long().size(0)
                        correct += (predicted == labels.long()).sum()
                    y_pred = np.array(y_pred)
                    y_labels = np.array(y_labels)
                    classification, confusion, results = auxil.reports(np.argmax(y_pred, axis=1), y_labels)
                    str_res = np.array2string(np.array(results), max_line_width=200)
                    print(str_res)

                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    # print(total)

                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')

                    # torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%, allresults= %s" % (epoch + 1, acc, str_res))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("results/best_acc_3resnet_S100_HSI.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
                        torch.save(net, 'results/model3_resnet_S100_HSI_model_best.pkl')
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
