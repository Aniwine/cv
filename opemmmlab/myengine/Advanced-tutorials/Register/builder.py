from mmengine.registry import Registry
from mmengine import MODELS as MMENGINE_MODELS
#这个文件用于实现注册器对象的构建，为了构建成功可以使用location字段，指定被注册模块所在位置，
# 也可以不指定，在使用时手动导入被注册模块，或是在此文件夹直接注册模块，或是继承机制，或是custom_import机制共四种

"""
0:通用流程
#1.创建注册器对象
#2.创建一个构建实例的函数(可选,保持默认即可),build_func
#3.使用注册器注册模块，如Register/activations/my_activation.py下实现的Sigmoid类
"""

"""
1:使用location字段
# 第一个参数是注册器的名字
# scope 表示注册器的作用域，如果不设置，默认为包名，例如在 mmdetection 中，它的 scope 为 mmdet
# locations 表示注册在此注册器的模块所存放的位置，注册器会根据预先定义的位置在构建模块时自动 import
#注意：location这种导入模块文件的方式，需要保证模块文件在已安装的模块的目录下，比如mmengine下，自己新建一个是会失败的
#这里假设在mmengine.models下新建了activations.py并将自定义类实现在此

ACTIVATION=Registry('activations',scope='mmengine',locations='mmengine.models.activations')
"""


"""
2:不指定location，在当前文件中进行注册
这样直接将注册模块放在同一个文件夹，也不需要指定location字段，也不需要使用时手动导入，
不过一般没有这样组织代码的，一般这种方式常常用于注册某些函数

import torch.nn as nn
ACTIVATION=Registry('activations',scope='mmengine')
@ACTIVATION.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Sigmoid.forward')
        return x
"""


"""
3:不指定location，使用时导入
"""

"""
4:不指定location，使用custom_import机制
"""
        


def build_activation(cfg,register,*args,**kwargs):
    cfg_=cfg.copy()
    act_type=cfg_.pop('type')
    print(f"build activation: {act_type}")
    act_cls=register.get(act_type)#register根据模块字符串，取得在此注册器中注册的该模块
    act=act_cls(*args,**kwargs,**cfg_)
    return act #返回构建的对象或者函数


#5 继承机制（最实用），继承mmengine的22个跟注册器，实现跨项目调用
# 参见https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/registry.html#id3
"""注册器"""
ACTIVATIONS=Registry('activations',scope='register',parent=MMENGINE_MODELS)
ACTIVATIONS2=Registry('activations',scope='config',parent=MMENGINE_MODELS)

#如果将被注册模块单独写到其他地方，然后构建注册器对象(ACTIVATIONS)时传入locations来指定位置，
#无论如何都要报错，暂时先将被注册模块和注册器对象放在同一文件夹下，如下：
#采用这种方式有两个缺点：1).文件不好组织，2).需要单独运行该builder.py文件触发，否则改动不生效
#如果测试用jupyter笔记本时还需要重启内核
"""被注册对象"""
"""激活函数类"""
import torch.nn as nn
#注册方式一：装饰器
@ACTIVATIONS.register_module()
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print('call Sigmoid.forward')
        return x
    
@ACTIVATIONS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call register LogSoftmax.forward')
        return x

@MMENGINE_MODELS.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call mmengine LogSoftmax.forward')
        return x
    
@ACTIVATIONS2.register_module()
class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()

    def forward(self, x):
        print('call config LogSoftmax.forward')
        return x