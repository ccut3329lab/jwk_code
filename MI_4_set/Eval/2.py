import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Load_Dataset.xxtset import Train_Data, Test_Data, sig_train, sig_test, label_train, label_test
from Models.Test_model import AMCNN_xxtset_trails

train_set = Train_Data(sig_train, label_train)
test_set = Test_Data(sig_test, label_test)
# train_loader = DataLoader(dataset=train_set, batch_size=394, shuffle=True, drop_last=True)
# test_loader = DataLoader(dataset=test_set, batch_size=98, shuffle=False, drop_last=True)

train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, drop_last=True)
# 加载训练好的模型
# model = EEGNet()
model = AMCNN_xxtset_trails()
model = model.type(torch.FloatTensor)
model.load_state_dict(torch.load('best_model2.pth'))


# 加载数据集


# 在模型上运行数据集，以获取模型的输出
model.eval()
for (sig_test, label_test) in test_loader:
    sig_test = sig_test
    label_test = label_test
with torch.no_grad():
    outputs = model(torch.tensor(sig_test).float())
    predictions = torch.argmax(outputs, dim=1)


# tsne = TSNE(n_components=2, perplexity=50, early_exaggeration=1.0, learning_rate=1000)   # perplexity=40, learning_rate=600,batch_size=128
tsne = TSNE(n_components=2, perplexity=40, early_exaggeration=1.0, learning_rate=600)   # perplexity=40, learning_rate=600,batch_size=128


# tsne = TSNE(n_components=2, perplexity=20, early_exaggeration=1.0, learning_rate=600)   # perplexity=50, learning_rate=600,batch_size=128



# tsne = TSNE(n_components=2, perplexity=70, early_exaggeration=10.0, learning_rate=600)   # perplexity=70, learning_rate=600,batch_size=128
# tsne = TSNE(n_components=2, perplexity=40, early_exaggeration=12.0, learning_rate=600)   # perplexity=40, learning_rate=600,batch_size=128
# tsne = TSNE(n_components=2, perplexity=90, learning_rate=600)
# tsne = TSNE(n_components=2, perplexity=90, early_exaggeration=12.0, learning_rate=200.0,
#             n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean',
#             init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None,
#             square_distances='legacy')







features = tsne.fit_transform(outputs.cpu().numpy())
colors = {'Class 0': 'red', 'Class 1': 'blue', 'Class 2': 'green', 'Class 3': 'orange'}


# 绘制可视化结果
fig, ax = plt.subplots()
for label in np.unique(label_test):
    ax.scatter(features[label_test == label, 0], features[label_test == label, 1], c=colors['Class {}'.format(label)], label='Class {}'.format(label))

ax.set_title('t-SNE Visualization of Model Outputs')
ax.legend()
plt.show()



# # 使用 t-SNE 将模型的输出映射到二维空间
# # tsne = TSNE(n_components=2, perplexity=70, learning_rate=600)
# tsne = TSNE(n_components=2, perplexity=90, learning_rate=600)
# features = tsne.fit_transform(outputs.cpu().numpy())
#
# colors = {'Class 0': 'red', 'Class 1': 'blue', 'Class 2': 'green', 'Class 3': 'orange'}
#
# # 绘制可视化结果
# for label in np.unique(label_test):
#     plt.scatter(features[label_test == label, 0], features[label_test == label, 1], c=colors['Class {}'.format(label)], label='Class {}'.format(label))
#
# # 绘制可视化结果
# plt.scatter(features[:, 0], features[:, 1], c=label_test)
# plt.title('t-SNE Visualization of Model Outputs')
# plt.show()


# tsne = TSNE(n_components=2, perplexity=90, learning_rate=600)
# features = tsne.fit_transform(outputs.cpu().numpy())
# colors = {'Class 0': 'red', 'Class 1': 'blue', 'Class 2': 'green', 'Class 3': 'orange'}
#
# # 绘制可视化结果
# for label in np.unique(label_test):
#     plt.scatter(features[label_test == label, 0], features[label_test == label, 1], c=colors['Class {}'.format(label)], label='Class {}'.format(label))
#
#     # 针对每个数据点，在散点图上添加文本标签
#     for i, point in enumerate(features):
#         if label_test[i] == label:
#             plt.annotate(str(i), xy=(point[0], point[1]), fontsize=8, color=colors['Class {}'.format(label)])
#
# plt.title('t-SNE Visualization of Model Outputs')
# plt.legend()
# plt.show()



