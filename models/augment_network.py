import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class SingleLayer(nn.Module):
    """
    单层卷积神经网络模块。

    Args:
        inc (int): 卷积层输入通道数，默认为32
        ouc (int): 卷积层输出通道数，默认为3
        kernel_size (int): 卷积核大小，默认为3
        tanh (bool): 是否使用Tanh激活函数，默认为False
        sig (bool): 是否使用Sigmoid激活函数，默认为False

    Attributes:
        conv1 (nn.Conv2d): 第一个卷积层
        conv2 (nn.Conv2d): 第二个卷积层
        tanh (bool): 是否使用Tanh激活函数
        sig (bool): 是否使用Sigmoid激活函数
    """

    def __init__(self, inc=32, ouc=3, kernel_size=3, tanh=False, sig=False):
        """
        初始化单层卷积神经网络模块。

        Args:
            inc (int): 卷积层输入通道数，默认为32
            ouc (int): 卷积层输出通道数，默认为3
            kernel_size (int): 卷积核大小，默认为3
            tanh (bool): 是否使用Tanh激活函数，默认为False
            sig (bool): 是否使用Sigmoid激活函数，默认为False
        """
        super(SingleLayer, self).__init__()
        # 第一个卷积层，将输入通道数转换为inc
        self.conv1 = nn.Conv2d(3, inc, kernel_size, 1, kernel_size // 2)
        # 第二个卷积层，将通道数转换为ouc
        self.conv2 = nn.Conv2d(inc, ouc, kernel_size, 1, kernel_size // 2)
        self.tanh = tanh  # 是否使用Tanh激活函数
        self.sig = sig  # 是否使用Sigmoid激活函数

    def forward(self, x):
        """
        前向传播过程。

        Args:
            x (Tensor): 输入张量

        Returns:
            Tensor: 输出张量
        """
        # 通过第一个卷积层
        x = self.conv1(x)
        # 通过第二个卷积层
        x = self.conv2(x)
        # 如果设置使用Tanh激活函数，应用Tanh
        if self.tanh:
            x = nn.Tanh()(x)
        # 如果设置使用Sigmoid激活函数，应用Sigmoid
        if self.sig:
            x = nn.Sigmoid()(x)
        return x