from builder import ACTIVATIONS



import torch.nn as nn

#注册方式一：装饰器
@ACTIVATIONS.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Sigmoid.forward')
        return x
    
#注册方式二：
"""
ACTIVATIONS.register_module(module=Sigmoid)
"""

@ACTIVATIONS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call LogSoftmax.forward')
        return x