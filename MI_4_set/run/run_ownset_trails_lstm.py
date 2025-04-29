import numpy as np
import torch
from pymanopt.manifolds import Sphere
from torch import nn, device, mean
from torch.utils.data import DataLoader
# from Load_Dataset.jwkset_lstm import Train_Data, Test_Data, sig_train, sig_test, label_train, label_test
from Load_Dataset.xxtset_lstm import Train_Data, Test_Data, sig_train, sig_test, label_train, label_test
from Models.Lstm_model import LSTM
from Models.Test_model import EEGNet, AMCNN_ownset_trails

# 超参设置
# batch_size = 8
batch_size = 16
# learning_rate = 1e-1  # 使用sgd
learning_rate = 1e-3    # 使用adm
# num_epochs = 300
num_epochs = 100
# num_epochs = 100
input_size = 501
hidden_size = 64
num_layers = 4
num_classes = 4


# 加载数据集
train_set = Train_Data(sig_train, label_train)
test_set = Test_Data(sig_test, label_test)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True)

# 设置损失和优化器
# net = CNN_LSTM()
# net = MultiBranchNet()
# net = MultiBranchCNN()
# net = EEGNet()                             # batch_size = 16   alpha=0.5  loss + 0.001 * net.l1_loss()
net = LSTM(input_size, hidden_size, num_layers, num_classes)                                # batch_size = 32
# net = AMCNN_ownset_1()                                # batch_size = 32
# net = CNN()
# net = net.cuda()
net = net.type(torch.cuda.FloatTensor)  # 数据集转成float.32
# net = net.type(torch.cuda.DoubleTensor)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  # 定义优化器，使用随机梯度下降
# 创建余弦退火调度器
# scheduler = CosineAnnealingLRWithRestart(optimizer, T_0=10, T_mult=2)
criterion = nn.CrossEntropyLoss()
# criterion = nn.MultiLabelSoftMarginLoss()
criterion = criterion.cuda()

# 开始模型训练
max_num = 0.0
total_num = 0.0
for epoch in range(num_epochs):
    print('*' * 10)
    print(f'epoch {epoch + 1}')
    running_loss = 0.0  # 初始值
    running_acc = 0.0

    # warmup_cosine(optimizer=optimizer, current_epoch=epoch, max_epoch=max_epoch, lr_min=lr_min, lr_max=lr_max,
    #               warmup_epoch=warmup_epoch)
    # print(optimizer.param_groups[0]['lr'])
    # cnt = 0
    # losssum = 0

    for i, (sig_train, label_train) in enumerate(train_loader, 1):  # 枚举函数enumerate返回下标和值
        best_valid_acc = np.inf
        # best_valid_loss = np.inf
        sig_train = sig_train.cuda()
        label_train = label_train.cuda()

        # 向前传播
        out = net(sig_train).cuda()  # 前向传播
        # loss = criterion(out, label_train)  # 计算loss
        loss = criterion(out, label_train.long())  # 计算loss
        # loss = criterion(out, label_train.long()) + 0.001 * net.l1_loss()  # 计算loss
        # loss = criterion(out, label_train.long()) + 0.001 * net.l2_loss()  # 计算loss
        # loss = criterion(out, label_train.long()) + 0.01 * net.l1_loss() + 0.001 * net.l2_loss()  # 计算loss
        running_loss += loss.item()  # loss求和
        _, pred = torch.max(out, 1)
        running_acc += (pred == label_train).float().mean()
        # 向后传播
        optimizer.zero_grad()  # 梯度归零
        # loss.requires_grad_(True)
        loss.backward()  # 后向传播
        optimizer.step()  # 更新参数
        # scheduler.step()

        if i % 300 == 0:
            print(f'[{epoch + 1}/{num_epochs}] Loss: {running_loss / i:.6f}, Acc: {running_acc / i:.6f}')
    print(f'Finish {epoch + 1} epoch, Loss: {running_loss / i:.6f}, Acc: {running_acc / i:.6f}')

    ## 模型测试
    # net.eval()  # 让模型变成测试模式
    eval_loss = 0.0
    eval_acc = 0.0
    for (sig_test, label_test) in test_loader:
        sig_test = sig_test.cuda()
        label_test = label_test.cuda()
        with torch.no_grad():
            out = net(sig_test).cuda()
            # loss = criterion(out, label_test)
            loss = criterion(out, label_test.long())
            # loss = criterion(out, label_test.long()) + 0.001 * net.l1_loss()  # 计算loss
            # loss = criterion(out, label_test.long()) + 0.001 * net.l2_loss()  # 计算loss
            # loss = criterion(out, label_test.long()) + 0.001 * net.l1_loss() + 0.001 * net.l2_loss()  # 计算loss
        eval_loss += loss.item()
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label_test).float().mean()
    print(f'Test Loss: {eval_loss / len(test_loader):.6f}, Acc: {eval_acc / len(test_loader):.6f}\n')

    # if eval_loss < best_valid_loss:
    #     best_valid_loss = eval_loss
    #     torch.save(net.state_dict(), 'best_model.pth')
    #     print("Saved the best model!")

    if eval_acc < best_valid_acc:
        best_valid_acc = eval_acc
        # torch.save(net.state_dict(), 'best_model1.pth')
        print("Saved the best model!")

    if eval_acc / len(test_loader) > max_num:
        max_num = eval_acc / len(test_loader)

    # total_num += eval_acc/ len(test_loader)

print('Best_acc:{}'.format(max_num))