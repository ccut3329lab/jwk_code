import torch
import torch.nn as nn  # import modules
from torch import cos, relu, sin
from torch.nn.functional import elu
from torch.nn.parameter import Parameter  # import Parameter to create custom activations with learnable parameters

from Act.act import Swish, Hexpo



def coselu1(x, a=0.1, b=0.5, c=1.0, d=1.0):
    y = d * (x + a * cos(b * x)) * elu(c * x)
    return y

def coselu_1(x, a=0.1, b=0.5, d=1.0):
    y = d * (2 * x + a * (cos(b * x) - b * sin(b * x)))
    return y

def coselu2(x, a=0.1, b=0.5, c=1.0, d=1.0):
    y = d * (x + a * cos(b * x)) * elu(c * x)
    return y

def coselu_2(x, a=0.1, b=0.5, d=1.0):
    y = d * (2 * x + a * (cos(b * x) - b * sin(b * x)))
    return y



# def coselu_1(x, a = 0.1, b = 0.5,c = 1.0,d = 1.0):
#     y = d*(x - a * cos(b * x)) * elu(c * x)
#     return y
#
# def coselu_2(x, a=0.1, b=0.5, d=1.0):
#     y = d * (2 * x - a * (cos(b * x) - b * sin(b * x)))
#     return y


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

        alpha_coselu1 = coselu1(self.alpha_t, a=0.5, b=1, c=1, d=0.5)              # alpha = 0.5  Best_acc:0.8576388955116272   batch_size = 32
        alpha_coselu_d1 = coselu1(alpha_coselu1, a=0.5, b=1, d=0.5).to(device)

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
        x_act = coselu1(self.alpha_t, a=0.5, b=1, c=1, d=0.5)
        # x_act = coselu(self.alpha_t, a=0.1, b=1, c=1, d=1)

        # x_act = coselu(self.alpha_t, a=-0.1, b=1, c=1, d=1.0)

        # 第二种激活函数
        # x_act = coselu(self.alpha_t, a=0.1, b=0.5, c=1, d=1.0)

        x = x_alpha.mul(alpha_mask) + x_act.mul(act_mask)

        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.Activation = nn.ELU()
        # self.Activation = nn.ReLU()
        # self.Activation = nn.Tanh()
        # self.Activation = nn.LeakyReLU()
        # self.Activation = nn.PReLU()
        # self.Activation = nn.Softplus()
        # self.Activation = nn.Hardswish()
        # self.Activation = Swish()
        # self.Activation = Hexpo()
        self.Activation = CosELU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # print(h0.shape)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # print(c0.shape)
        out, _ = self.lstm(x, (h0, c0))
        # print(out.shape)
        out = self.Activation(out)
        out = self.fc(out[:, -1, :])
        # print(out.shape)
        return out

if __name__ == '__main__':
    input_size = 501
    hidden_size = 64
    num_layers = 5
    num_classes = 4

    model = LSTM(input_size, hidden_size, num_layers, num_classes).cuda()
    print(model)
    # inputs = torch.randn(64, 1, 8, 501).cuda()

    inputs = torch.randn(64, 1, 8, 501).cuda()
    inputs = inputs.squeeze(1)  # 去除第2个维度（大小为1）
    output = model(inputs)
    print(output.shape)