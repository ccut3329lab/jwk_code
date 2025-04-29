import torch
import torch.nn as nn  # import modules
from torch.nn.parameter import Parameter  # import Parameter to create custom activations with learnable parameters
import torch.nn.functional as F
from torch import cos, relu, sin
from torch.nn.functional import elu

class ReLTanh(nn.Module):
    '''
    Implementation of ReLTanh activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - constant parameter
        - beta - constant parameter
    References:
        - See related paper:
        https://www.sciencedirect.com/science/article/pii/S0925231219309464
    Examples:
        # >>> a1 = ReLTanh(0, -1.5)
        # >>> x = torch.randn(256)
        # >>> x = a1(x)
    '''

    def __init__(self, alpha=0.0, beta=-1.5, learnable=True):
        super(ReLTanh, self).__init__()

        assert alpha > beta
        # print(f'alpha : {alpha}')
        # print(f'beta : {beta}')
        self.alpha = alpha
        self.beta = beta
        self.alpha_t = torch.FloatTensor([self.alpha])
        self.beta_t = torch.FloatTensor([self.beta])
        # print(f'alpha_t : {self.alpha_t}')
        # print(f'beta_t : {self.beta_t}')

        self.learnable = learnable

        if self.learnable:
            self.alpha_t = Parameter(self.alpha_t, requires_grad=True)
            self.beta_t = Parameter(self.beta_t, requires_grad=True)
            # print(f'alpha_t : {self.alpha_t}')
            # print(f'beta_t : {self.beta_t}')

    def __repr__(self):
        if self.learnable:
            return f"ReLTanh(alpha={self.alpha_t.data.item()}, beta={self.beta_t.data.item()}, learnable={self.learnable})"
        return f"ReLTanh(alpha={self.alpha}, beta={self.beta})"

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        # move variables to correct device
        device = x.device
        self.alpha_t = self.alpha_t.to(device)
        self.beta_t = self.beta_t.to(device)

        # Set up boundary values
        alpha_tanh = torch.tanh(self.alpha_t)
        beta_tanh = torch.tanh(self.beta_t)
        # print(f'alpha_tanh: {alpha_tanh}')
        # print(f'beta_tanh: {beta_tanh}')
        one = torch.ones([1]).to(device)
        # print(f'one: {one }')
        alpha_tanh_d1 = one.sub(torch.square(alpha_tanh))
        beta_tanh_d1 = one.sub(torch.square(beta_tanh))
        # print(f'alpha_tanh_d1: {alpha_tanh_d1}')
        # print(f'beta_tanh_d1: {beta_tanh_d1 }')

        # compute masks to relax indifferentiability
        alpha_mask = x.ge(self.alpha_t)
        beta_mask = x.le(self.beta_t)
        # print(f'alpha_mask: {alpha_mask}')
        # print(f'beta_mask : {beta_mask }')
        act_mask = ~(alpha_mask | beta_mask)
        # print(f'act_mask: {act_mask}')

        # activations
        x_alpha = x.sub(self.alpha_t).mul(alpha_tanh_d1).add(self.alpha_t)
        x_beta = x.sub(self.beta_t).mul(beta_tanh_d1).add(self.beta_t)
        x_act = torch.tanh(x)

        # combine activations
        x = x_alpha.mul(alpha_mask) + x_beta.mul(beta_mask) + x_act.mul(act_mask)

        return x


def coselu(x, a=0.1, b=0.5, c=2.0, d=1.0):
    y = d * (x + a * cos(b * x)) * relu(c * x)
    return y


class CosELU(nn.Module):
    '''
    Implementation of ReLTanh activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - constant parameter
        - beta - constant parameter
    References:
        - See related paper:
        https://www.sciencedirect.com/science/article/pii/S0925231219309464
    Examples:
        # >>> a1 = ReLTanh(0, -1.5)
        # >>> x = torch.randn(256)
        # >>> x = a1(x)
    '''

    def __init__(self, alpha=0.58, learnable=True):   # EEGNet: alpha = 0.5    a=0.1, b=1, c=1, d=0.1  Softmax  0.58
        super(CosELU, self).__init__()               # AMCNN: alpha = 0.0    a=0.1, b=1, c=1, d=0.1

        assert alpha >= 0
        # print(f'alpha : {alpha}')
        # print(f'beta : {beta}')
        self.alpha = alpha
        self.alpha_t = torch.FloatTensor([self.alpha])
        # print(f'alpha_t : {self.alpha_t}')
        # print(f'beta_t : {self.beta_t}')

        self.learnable = learnable

        if self.learnable:
            self.alpha_t = Parameter(self.alpha_t, requires_grad=True)
            # print(f'alpha_t : {self.alpha_t}')
            # print(f'beta_t : {self.beta_t}')

    def __repr__(self):
        if self.learnable:
            return f"CosELU(alpha={self.alpha_t.data.item()}, learnable={self.learnable})"
        return f"CosELU(alpha={self.alpha})"

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        # move variables to correct device
        device = x.device
        # print(device)
        self.alpha_t = self.alpha_t.to(device)
        # print(coselu(x, a=0.2, b=1, c=1, d=0.1))
        # print(self.alpha_t)
        # Set up boundary values
        # alpha_coselu = coselu(self.alpha_t, a=0.2, b=1, c=1, d=0.1) # 原始
        alpha_coselu = coselu(self.alpha_t, a=0.1, b=1, c=1, d=0.1) # 原始   1
        # alpha_coselu = coselu(self.alpha_t, a=0.5, b=1, c=1, d=0.5)     #   2
        # alpha_coselu = coselu(self.alpha_t, a=0.1, b=1, c=1, d=1)      # 3

        # alpha_coselu = coselu(self.alpha_t, a=0.01, b=1, c=1, d=0.01)  # 原始   best
        # alpha_coselu = coselu(self.alpha_t, a=0.01, b=1, c=1, d=0.5)  # 原始   best1
        # alpha_coselu = coselu(self.alpha_t, a=0.01, b=1, c=1, d=0.25)  # 原始   best2
        # alpha_coselu = coselu(self.alpha_t, a=0.01, b=1, c=1, d=0.005)  # 原始   best2



        # alpha_coselu = coselu(self.alpha_t, a=1/3, b=1, c=1, d=3)

        # alpha_coselu = coselu(self.alpha_t, a =1, b = 1,c = 1,d=0.1)
        # alpha_coselu = coselu(self.alpha_t, a =1, b = 1,c = 1,d=0.1)
        # alpha_coselu = coselu(self.alpha_t, a =2, b = 0.1,c = 1,d=0.1)
        # print(f'alpha_tanh: {alpha_tanh}')
        # print(f'beta_tanh: {beta_tanh}')
        one = torch.ones([1]).to(device)
        # print(f'one: {one }')
        alpha_coselu_d1 = one.sub(torch.square(alpha_coselu))
        # print(f'alpha_tanh_d1: {alpha_tanh_d1}')
        # print(f'beta_tanh_d1: {beta_tanh_d1 }')

        # compute masks to relax indifferentiability
        alpha_mask = x.ge(self.alpha_t)
        # beta_mask = x.le(self.beta_t)
        # print(f'alpha_mask: {alpha_mask}')
        # print(f'beta_mask : {beta_mask }')
        act_mask = ~(alpha_mask)
        # print(f'act_mask: {act_mask}')

        # ~ x就是 - (x + 1)

        # activations
        x_alpha = x.sub(self.alpha_t).mul(alpha_coselu_d1).add(self.alpha_t)
        x_act = coselu(self.alpha_t, a=0.1, b=1, c=1, d=0.1)  # 1
        # x_act = coselu(self.alpha_t, a=0.5, b=1, c=1, d=0.5)  # 2
        # x_act = coselu(self.alpha_t, a=0.1, b=1, c=1, d=1)  # 3

        # x_act = coselu(self.alpha_t, a=0.01, b=1, c=1, d=0.01)  # best
        # x_act = coselu(self.alpha_t, a=0.01, b=1, c=1, d=0.5)  # best1
        # x_act = coselu(self.alpha_t, a=0.01, b=1, c=1, d=0.25)  # best2
        # x_act = coselu(self.alpha_t, a=0.01, b=1, c=1, d=0.005)  # best3

        # x_act = coselu(self.alpha_t, a=1/3, b=1, c=1, d=3)

        # combine activations
        x = x_alpha.mul(alpha_mask) + x_act.mul(act_mask)

        return x

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


class CosELU2(nn.Module):

    def __init__(self, alpha=0.0, learnable=True):                                   # 1: 0.5 < = alpha < = 1.0       EEGNet_best: alpha = 0.5                              AMCNN_best: alpha = 0.8
        super(CosELU2, self).__init__()  # 2: 0.8 < = alpha < = 1.3                   EEGNet_best:  batch_size = 16   alpha = 0.8    batch_size = 32  alpha = 0.0            AMCNN_best: alpha = 1.0
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





class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)

class Hexpo(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(Hexpo, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        return torch.where(x < 0, self.alpha * (torch.exp(x) - 1), self.beta * x)

if __name__ == '__main__':
    # a1 = ReLTanh(0, -1.5)
    # x = torch.randn(256)
    # x = a1(x)
    # print(a1)

    a1 = CosELU(10.0)
    x = torch.randn(256)
    x = a1(x)
    print(a1)

    # model = ReLTanh()
    # x = torch.randn(256)
    # print(model(x))
