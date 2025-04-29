import torch
import torch.nn as nn  # import modules
from torch import cos, relu, sin
from torch.nn.functional import elu
from torch.nn.parameter import Parameter  # import Parameter to create custom activations with learnable parameters

from Act.act import Swish, Hexpo
from Models.ECANet import eca_block


def coselu(x, a=0.1, b=0.5, c=1.0, d=1.0):
    y = d * (x + a * cos(b * x)) * elu(c * x)
    return y

def coselu_1(x, a = 0.1, b = 0.5,c = 1.0,d = 1.0):
    y = d*(x - a * cos(b * x)) * elu(c * x)
    return y

def coselu1(x, a=0.1, b=0.5, d=1.0):
    y = d * (2 * x + a * (cos(b * x) - b * sin(b * x)))
    return y

def coselu_2(x, a=0.1, b=0.5, d=1.0):
    y = d * (2 * x - a * (cos(b * x) - b * sin(b * x)))
    return y


class CosELU(nn.Module):

    def __init__(self, alpha=0.0, learnable=True):                                   # 1: 0.5 < = alpha < = 1.0       EEGNet_best: alpha = 0.5                              AMCNN_best: alpha = 0.8
        super(CosELU, self).__init__()  # 2: 0.8 < = alpha < = 1.3                   EEGNet_best:  batch_size = 16   alpha = 0.8    batch_size = 32  alpha = 0.0            AMCNN_best: alpha = 1.0
                                                                                     # 3: 0.0 < = alpha < = 1.0    best: alpha = 0.5
        assert alpha >= 0
        self.alpha = alpha
        self.alpha_t = torch.FloatTensor([self.alpha])

        self.learnable = learnable

        if self.learnable:
            self.alpha_t = Parameter(self.alpha_t, requires_grad=True)

    def __repr__(self):
        if self.learnable:
            return f"CosELU(alpha={self.alpha_t.data.item()}, learnable={self.learnable})"
        return f"CosELU(alpha={self.alpha})"

    def forward(self, x):
        # move variables to correct device
        device = x.device
        self.alpha_t = self.alpha_t.to(device)
        # Set up boundary values
        # alpha_coselu = coselu(self.alpha_t, a=0.1, b=1, c=1, d=0.1)            # alpha = 1.0  Best_acc:0.8472222089767486    batch_size = 32
        # alpha_coselu_d1 = coselu1(alpha_coselu, a=0.1, b=1, d=0.1).to(device)

        alpha_coselu = coselu(self.alpha_t, a=0.5, b=1, c=1, d=0.5)              # alpha = 0.5  Best_acc:0.8576388955116272   batch_size = 32
        alpha_coselu_d1 = coselu1(alpha_coselu, a=0.5, b=1, d=0.5).to(device)

        # alpha_coselu = coselu(self.alpha_t, a=0.1, b=1, c=1, d=1)              # alpha = 0.5  Best_acc:0.81944  Mean_acc:0.71865
        # alpha_coselu_d1 = coselu1(alpha_coselu, a=0.1, b=1, d=1).to(device)

        # alpha_coselu = coselu(self.alpha_t, a=-0.1, b=1, c=1, d=1.0)
        # alpha_coselu_d1 = coselu1(alpha_coselu, a=-0.1, b=1, d=1.0).to(device)

        # # 第二种激活函数
        # alpha_coselu = coselu_1(self.alpha_t,  a=0.1, b=0.5, c=1.0, d=1.0)
        # alpha_coselu_d1 = coselu_2(alpha_coselu, a=0.1, b=0.5, d=1.0).to(device)




        # compute masks to relax indifferentiability
        alpha_mask = x.ge(self.alpha_t)
        act_mask = ~alpha_mask
        x_alpha = x.sub(self.alpha_t).mul(alpha_coselu_d1).add(self.alpha_t)
        # x_act = coselu(self.alpha_t, a=0.1, b=1, c=1, d=0.1)
        x_act = coselu(self.alpha_t, a=0.5, b=1, c=1, d=0.5)
        # x_act = coselu(self.alpha_t, a=0.1, b=1, c=1, d=1)

        # x_act = coselu(self.alpha_t, a=-0.1, b=1, c=1, d=1.0)

        # 第二种激活函数
        # x_act = coselu(self.alpha_t, a=0.1, b=0.5, c=1, d=1.0)

        x = x_alpha.mul(alpha_mask) + x_act.mul(act_mask)

        return x

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              groups=in_channels)

    def forward(self, x):
        return self.conv(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class EEGNet(nn.Module):
    def __init__(self, n_classes=4, channels=22, samples=1000,
                 dropoutRate=0.5, kernelLength=64, F1=8,
                 D=2, F2=16):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.dropoutRate = dropoutRate

        # Conv2D =======================
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=0, bias=False)
        self.BatchNorm2d1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        # self.BatchNorm2d1 = nn.BatchNorm2d(self.F1, False)

        # DepthwiseConv2D =======================
        self.conv2 = DepthwiseConv2d(self.F1, self.F1 * self.D, (self.channels, 1), stride=1, padding=0)
        self.BatchNorm2d2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
        # self.BatchNorm2d2 = nn.BatchNorm2d(self.F1 * self.D, False)
        self.relu2 = CosELU()
        # self.relu2 = nn.ELU()
        # self.relu2 = nn.ReLU()
        # self.relu2 = nn.Tanh()
        # self.relu2 = nn.LeakyReLU()
        # self.relu2 = nn.PReLU()
        # self.relu2 = nn.Softplus()
        # self.relu2 = nn.Hardswish()
        # self.relu2 = Swish()
        # self.relu2 = Hexpo()
        self.pooling2 = nn.AvgPool2d((1, 4))
        self.dropout2 = nn.Dropout(dropoutRate)

        # SeparableConv2D =======================.
        self.conv3 = SeparableConv2d(self.F1 * self.D, self.F2, (1, self.kernelLength // 2), stride=1,
                                     padding=0)  # padding=1
        self.BatchNorm2d3 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        # self.BatchNorm2d3 = nn.BatchNorm2d(self.F1, False)
        self.relu3 = CosELU()
        # self.relu3 = nn.ELU()
        # self.relu3 = nn.ReLU()
        # self.relu3 = nn.Tanh()
        # self.relu3 = nn.LeakyReLU()
        # self.relu3 = nn.PReLU()
        # self.relu3 = nn.Softplus()
        # self.relu3 = nn.Hardswish()
        # self.relu3 = Swish()
        # self.relu3 = Hexpo()

        self.pooling3 = nn.AvgPool2d((1, 8))
        self.dropout3 = nn.Dropout(dropoutRate)
        self.flatten = nn.Flatten()
        # Fc层
        self.fc = nn.Linear(16 * 1 * 25, n_classes)  # 16*3*25
        # self.relu4 = nn.Sigmoid()
        self.relu4 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BatchNorm2d1(x)

        x = self.conv2(x)
        x = self.BatchNorm2d2(x)
        x = self.relu2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.BatchNorm2d3(x)
        x = self.relu3(x)
        x = self.pooling3(x)
        x = self.dropout3(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu4(x)

        return x

    def l1_loss(self, alpha=0.001):
        loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                loss += alpha * torch.sum(torch.abs(param))
        return loss

    def l2_loss(self, alpha=0.001):
        loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                loss += alpha * torch.sum(torch.pow(param, 2))
        return loss


class AMCNN_ownset_trails(nn.Module):
    def __init__(self, n_classes=4, channels=6, samples=1001,
                 dropoutRate=0.5, kernelLength=64, F1=8,
                 D=2, F2=16):
        super(AMCNN_ownset_trails, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.dropoutRate = dropoutRate

        # Conv2D =======================
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=0, bias=False)
        self.BatchNorm2d1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.ECANet = eca_block()

        # DepthwiseConv2D =======================
        self.conv2 = DepthwiseConv2d(self.F1, self.F1 * self.D, (self.channels, 1), stride=1, padding=0)
        self.BatchNorm2d2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
        # self.relu2 = ModifiedReLU(1.0)
        # self.relu2 = CosELU()
        # self.relu2 = nn.ELU()
        # self.relu2 = nn.ReLU()
        # self.relu2 = nn.Tanh()
        # self.relu2 = nn.LeakyReLU()
        # self.relu2 = nn.PReLU()
        # self.relu2 = nn.Softplus()
        # self.relu2 = nn.Hardswish()
        # self.relu2 = Swish()
        self.relu2 = Hexpo()
        self.pooling2 = nn.AvgPool2d((1, 2))
        self.dropout2 = nn.Dropout(dropoutRate)

        # SeparableConv2D =======================.
        self.conv3 = SeparableConv2d(self.F1 * self.D, self.F2, (1, self.kernelLength), stride=1, padding=0)
        self.BatchNorm2d3 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        # self.relu3 = ModifiedReLU(1.0)
        # self.relu3 = CosELU()
        # self.relu3 = nn.ELU()
        # self.relu3 = nn.ReLU()
        # self.relu3 = nn.Tanh()
        # self.relu3 = nn.LeakyReLU()
        # self.relu3 = nn.PReLU()
        # self.relu3 = nn.Softplus()
        # self.relu3 = nn.Hardswish()
        # self.relu3 = Swish()
        self.relu3 = Hexpo()
        self.pooling3 = nn.AvgPool2d((1, 4))
        self.dropout3 = nn.Dropout(dropoutRate)

        # Conv2D =======================.
        self.conv4 = nn.Conv2d(self.F2, self.F2, (1, self.kernelLength // 2), stride=1, padding=0,
                               bias=False)  # padding=1
        self.BatchNorm2d4 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        # self.relu4 = ModifiedReLU(1.0)
        # self.relu4 = CosELU()
        # self.relu4 = nn.ELU()
        # self.relu4 = nn.ReLU()
        # self.relu4 = nn.Tanh()
        # self.relu4 = nn.LeakyReLU()
        # self.relu4 = nn.PReLU()
        # self.relu4 = nn.Softplus()
        # self.relu4 = nn.Hardswish()
        # self.relu4 = Swish()
        self.relu4 = Hexpo()
        self.pooling4 = nn.AvgPool2d((1, 8))
        self.dropout4 = nn.Dropout(dropoutRate)

        self.flatten = nn.Flatten()
        # Fc层
        # self.fc = nn.Linear(16 * 1 * 8, n_classes)
        # self.fc = nn.Linear(16 * 1 * 1, n_classes)
        self.fc = nn.Linear(16 * 3 * 1, n_classes)
        self.relu5 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # print('x1:', x.shape)
        x = self.BatchNorm2d1(x)

        x = self.ECANet(x)

        x = self.conv2(x)
        x = self.BatchNorm2d2(x)
        x = self.relu2(x)
        # print(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        # print('x3:', x.shape)
        x = self.BatchNorm2d3(x)
        x = self.relu3(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        # print('x4:', x.shape)
        x = self.BatchNorm2d4(x)
        x = self.relu4(x)
        x = self.pooling4(x)
        x = self.dropout4(x)
        # print('x5:', x.shape)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu5(x)
        # print('x6:', x.shape)

        return x

    def l1_loss(self, alpha=0.001):
        loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                loss += alpha * torch.sum(torch.abs(param))
        return loss

    def l2_loss(self, alpha=0.001):
        loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                loss += alpha * torch.sum(torch.pow(param, 2))
        return loss

class AMCNN_xxtset_trails(nn.Module):
    def __init__(self, n_classes=4, channels=6, samples=1001,
                 dropoutRate=0.5, kernelLength=64, F1=8,
                 D=2, F2=16):
        super(AMCNN_xxtset_trails, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.dropoutRate = dropoutRate

        # Conv2D =======================
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=0, bias=False)
        self.BatchNorm2d1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.ECANet = eca_block()

        # DepthwiseConv2D =======================
        self.conv2 = DepthwiseConv2d(self.F1, self.F1 * self.D, (self.channels, 1), stride=1, padding=0)
        self.BatchNorm2d2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
        # self.relu2 = ModifiedReLU(1.0)
        # self.relu2 = CosELU()
        # self.relu2 = nn.ELU()
        # self.relu2 = nn.ReLU()
        # self.relu2 = nn.Tanh()
        # self.relu2 = nn.LeakyReLU()
        # self.relu2 = nn.PReLU()
        # self.relu2 = nn.Softplus()
        # self.relu2 = nn.Hardswish()
        # self.relu2 = Swish()
        self.relu2 = Hexpo()
        self.pooling2 = nn.AvgPool2d((1, 2))
        self.dropout2 = nn.Dropout(dropoutRate)

        # SeparableConv2D =======================.
        self.conv3 = SeparableConv2d(self.F1 * self.D, self.F2, (1, self.kernelLength), stride=1, padding=0)
        self.BatchNorm2d3 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        # self.relu3 = ModifiedReLU(1.0)
        # self.relu3 = CosELU()
        # self.relu3 = nn.ELU()
        # self.relu3 = nn.ReLU()
        # self.relu3 = nn.Tanh()
        # self.relu3 = nn.LeakyReLU()
        # self.relu3 = nn.PReLU()
        # self.relu3 = nn.Softplus()
        # self.relu3 = nn.Hardswish()
        # self.relu3 = Swish()
        self.relu3 = Hexpo()
        self.pooling3 = nn.AvgPool2d((1, 4))
        self.dropout3 = nn.Dropout(dropoutRate)

        # Conv2D =======================.
        self.conv4 = nn.Conv2d(self.F2, self.F2, (1, self.kernelLength // 2), stride=1, padding=0,
                               bias=False)  # padding=1
        self.BatchNorm2d4 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        # self.relu4 = ModifiedReLU(1.0)
        # self.relu4 = CosELU()
        # self.relu4 = nn.ELU()
        # self.relu4 = nn.ReLU()
        # self.relu4 = nn.Tanh()
        # self.relu4 = nn.LeakyReLU()
        # self.relu4 = nn.PReLU()
        # self.relu4 = nn.Softplus()
        # self.relu4 = nn.Hardswish()
        # self.relu4 = Swish()
        self.relu4 = Hexpo()
        self.pooling4 = nn.AvgPool2d((1, 8))
        self.dropout4 = nn.Dropout(dropoutRate)

        self.flatten = nn.Flatten()
        # Fc层
        # self.fc = nn.Linear(16 * 1 * 8, n_classes)
        # self.fc = nn.Linear(16 * 1 * 1, n_classes)
        self.fc = nn.Linear(16 * 3 * 1, n_classes)
        self.relu5 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # print('x1:', x.shape)
        x = self.BatchNorm2d1(x)

        x = self.ECANet(x)

        x = self.conv2(x)
        x = self.BatchNorm2d2(x)
        x = self.relu2(x)
        # print(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        # print('x3:', x.shape)
        x = self.BatchNorm2d3(x)
        x = self.relu3(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        # print('x4:', x.shape)
        x = self.BatchNorm2d4(x)
        x = self.relu4(x)
        x = self.pooling4(x)
        x = self.dropout4(x)
        # print('x5:', x.shape)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu5(x)
        # print('x6:', x.shape)

        return x

    def l1_loss(self, alpha=0.001):
        loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                loss += alpha * torch.sum(torch.abs(param))
        return loss

    def l2_loss(self, alpha=0.001):
        loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                loss += alpha * torch.sum(torch.pow(param, 2))
        return loss

class AMCNN_patient1_trails(nn.Module):
    def __init__(self, n_classes=4, channels=6, samples=1001,
                 dropoutRate=0.5, kernelLength=64, F1=8,
                 D=2, F2=16):
        super(AMCNN_patient1_trails, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.dropoutRate = dropoutRate

        # Conv2D =======================
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=0, bias=False)
        self.BatchNorm2d1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.ECANet = eca_block()

        # DepthwiseConv2D =======================
        self.conv2 = DepthwiseConv2d(self.F1, self.F1 * self.D, (self.channels, 1), stride=1, padding=0)
        self.BatchNorm2d2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
        # self.relu2 = ModifiedReLU(1.0)
        self.relu2 = CosELU()
        # self.relu2 = nn.ELU()
        # self.relu2 = nn.ReLU()
        # self.relu2 = nn.Tanh()
        # self.relu2 = nn.LeakyReLU()
        # self.relu2 = nn.PReLU()
        # self.relu2 = nn.Softplus()
        # self.relu2 = nn.Hardswish()
        # self.relu2 = Swish()
        self.relu2 = Hexpo()
        self.pooling2 = nn.AvgPool2d((1, 2))
        self.dropout2 = nn.Dropout(dropoutRate)

        # SeparableConv2D =======================.
        self.conv3 = SeparableConv2d(self.F1 * self.D, self.F2, (1, self.kernelLength), stride=1, padding=0)
        self.BatchNorm2d3 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        # self.relu3 = ModifiedReLU(1.0)
        self.relu3 = CosELU()
        # self.relu3 = nn.ELU()
        # self.relu3 = nn.ReLU()
        # self.relu3 = nn.Tanh()
        # self.relu3 = nn.LeakyReLU()
        # self.relu3 = nn.PReLU()
        # self.relu3 = nn.Softplus()
        # self.relu3 = nn.Hardswish()
        # self.relu3 = Swish()
        # self.relu3 = Hexpo()
        self.pooling3 = nn.AvgPool2d((1, 4))
        self.dropout3 = nn.Dropout(dropoutRate)

        # Conv2D =======================.
        self.conv4 = nn.Conv2d(self.F2, self.F2, (1, self.kernelLength // 2), stride=1, padding=0,
                               bias=False)  # padding=1
        self.BatchNorm2d4 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        # self.relu4 = ModifiedReLU(1.0)
        self.relu4 = CosELU()
        # self.relu4 = nn.ELU()
        # self.relu4 = nn.ReLU()
        # self.relu4 = nn.Tanh()
        # self.relu4 = nn.LeakyReLU()
        # self.relu4 = nn.PReLU()
        # self.relu4 = nn.Softplus()
        # self.relu4 = nn.Hardswish()
        # self.relu4 = Swish()
        # self.relu4 = Hexpo()
        self.pooling4 = nn.AvgPool2d((1, 8))
        self.dropout4 = nn.Dropout(dropoutRate)

        self.flatten = nn.Flatten()
        # Fc层
        # self.fc = nn.Linear(16 * 1 * 8, n_classes)
        # self.fc = nn.Linear(16 * 1 * 1, n_classes)
        self.fc = nn.Linear(16 * 3 * 1, n_classes)
        self.relu5 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # print('x1:', x.shape)
        x = self.BatchNorm2d1(x)

        x = self.ECANet(x)

        x = self.conv2(x)
        x = self.BatchNorm2d2(x)
        x = self.relu2(x)
        # print(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        # print('x3:', x.shape)
        x = self.BatchNorm2d3(x)
        x = self.relu3(x)
        x = self.pooling3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        # print('x4:', x.shape)
        x = self.BatchNorm2d4(x)
        x = self.relu4(x)
        x = self.pooling4(x)
        x = self.dropout4(x)
        # print('x5:', x.shape)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu5(x)
        # print('x6:', x.shape)

        return x

    def l1_loss(self, alpha=0.001):
        loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                loss += alpha * torch.sum(torch.abs(param))
        return loss

    def l2_loss(self, alpha=0.001):
        loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                loss += alpha * torch.sum(torch.pow(param, 2))
        return loss


if __name__ == '__main__':
    # model = CNN().cuda()
    # model = AMCNN().cuda()
    # model = CNN_LSTM().cuda()
    # model = MultiBranchCNN()
    # model = MultiBranchNet().cuda()
    # model = AMCNN_ownset().cuda()
    # model = AMCNN_ownset_1().cuda()
    model = AMCNN_ownset_trails().cuda()
    print(model)
    # inputs = torch.randn(64, 1, 22, 1000).cuda()
    # inputs = torch.randn(64, 1, 8, 1001).cuda()
    inputs = torch.randn(64, 1, 8, 501).cuda()
    # inputs = torch.randn(64, 1, 8, 1503).cuda()
    # inputs = torch.randn(64, 1, 6, 1503).cuda()
    output = model(inputs)
    print(output.shape)

# if __name__ == '__main__':
#     model = EEGNet()
#     print(model)
#     input = torch.randn(288, 1, 22, 1000)
#     out = model(input)
#     print(out.shape)

# model = VggNet()
# print(model)
# input = torch.randn(288, 1, 22, 1000)
# out = model(input)
# print(out.shape)

# model = AMCNN().cuda()
# print(model)
# input = torch.randn(288, 1, 22, 1000).cuda()
# out = model(input)
# print(out.shape)

# model = ModifiedReLU(threshold=1.0)
# print(model)

# sig_train = sig_train.astype(np.float32)
# sig_test = sig_test.astype(np.float32)
