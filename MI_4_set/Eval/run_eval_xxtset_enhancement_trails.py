import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from Load_Dataset.xxtset import Train_Data, Test_Data, sig_train, sig_test, label_train, label_test
from Models.Test_model import AMCNN_xxtset_trails

# 超参设置
# batch_size = 4
# learning_rate = 1e-3
# num_epochs = 40

batch_size = 8
learning_rate = 1e-3
num_epochs = 300

# 加载数据集
train_set = Train_Data(sig_train, label_train)
test_set = Test_Data(sig_test, label_test)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True)

best_model = net = AMCNN_xxtset_trails()
best_model = best_model.type(torch.cuda.DoubleTensor)
best_model.load_state_dict(torch.load('best_model2.pth'))
best_model.eval()

# 定义一些变量
y_true = np.array([])
y_pred = np.array([])

# 遍历测试数据进行预测
for (sig_test, label_test) in test_loader:
    sig_test = sig_test.cuda()
    label_test = label_test
    label_test = label_test.numpy()
    y_true = np.concatenate([y_true, label_test])
    out = best_model(sig_test)
    _, predicted = torch.max(out, 1)
    predicted = predicted.cpu().numpy()
    y_pred = np.concatenate([y_pred, predicted])

matplotlib.rcParams['font.sans-serif'] = ['SimHei']

# new_labels = [左手', '右手', '左脚', '右脚']
new_labels = ['Left Hand', 'Right Hand', 'Left Foot', 'Right Foot']

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 计算预测准确率
accuracy = np.trace(cm) / np.sum(cm)

# 将混淆矩阵中的数据转换为百分比形式
# cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
cm_percent = cm / cm.sum(axis=1, keepdims=True)

# 将混淆矩阵中的数据转换为百分比形式字符串
cm_percent_str = [["{:.1f}%".format(val) for val in row] for row in cm_percent]

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
# sns.heatmap(cm_percent, annot=cm_percent_str, cmap="Reds", fmt="", cbar=True, xticklabels=new_labels, yticklabels=new_labels)
heatmap = sns.heatmap(cm_percent, annot=True, cmap="Reds", fmt=".1%", cbar=True, vmin=0, vmax=1, xticklabels=new_labels,yticklabels=new_labels)
# heatmap = sns.heatmap(cm_percent, annot=True, cmap="Blues", fmt=".1%", cbar=True, vmin=0, vmax=1, xticklabels=new_labels, yticklabels=new_labels)
plt.title(f"Confusion Matrix (Accuracy = {accuracy:.3f})")
# plt.xlabel("预测标签", fontsize=12, labelpad=10)
# plt.ylabel("真实标签", fontsize=12, labelpad=10)
plt.xlabel("Predicted Labels", fontsize=12, labelpad=10)
plt.ylabel("True Labels", fontsize=12, labelpad=10)
# 调整坐标轴刻度位置
plt.tick_params(axis='both', which='both', length=0, pad=10)

# 调整colorbar的位置
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(length=0, pad=10)
plt.show()















# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# from sklearn.metrics import confusion_matrix
# from torch.utils.data import DataLoader
# from Load_Dataset.dataset_test import Train_Data, Test_Data, sig_train, label_train, sig_test, label_test
# from Models.Test_model import AMCNN, EEGNet, MultiBranchCNN, MultiBranchNet, CNN, CNN_LSTM
#
# # 超参设置
# batch_size = 32
# learning_rate = 1e-3
# num_epochs = 300
#
# # 加载数据集
# train_set = Train_Data(sig_train, label_train)
# test_set = Test_Data(sig_test, label_test)
# train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
# test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True)
#
# best_model = AMCNN()
# best_model = best_model.type(torch.cuda.DoubleTensor)
# best_model.load_state_dict(torch.load('best_model.pth'))
# best_model.eval()
#
# # 定义一些变量
# y_true = np.array([])
# y_pred = np.array([])
#
# # 遍历测试数据进行预测
# for (sig_test, label_test) in test_loader:
#     sig_test = sig_test.cuda()
#     label_test = label_test
#     label_test = label_test.numpy()
#     y_true = np.concatenate([y_true, label_test])
#     out = best_model(sig_test)
#     _, predicted = torch.max(out, 1)
#     predicted = predicted.cpu().numpy()
#     y_pred = np.concatenate([y_pred, predicted])
#
# # 计算混淆矩阵
# cm = confusion_matrix(y_true, y_pred)
#
# # 计算预测准确率
# accuracy = np.trace(cm) / np.sum(cm)
#
# # 将混淆矩阵中的数据转换为百分比形式
# cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
#
# # 可视化混淆矩阵
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm_percent, annot=True, cmap="Reds", fmt=".2f", cbar=True)
# plt.title(f"Confusion Matrix (Accuracy = {accuracy:.4f})")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.show()








