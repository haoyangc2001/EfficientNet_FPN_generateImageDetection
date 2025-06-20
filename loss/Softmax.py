import torch
import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
    def __init__(self, config):
        super(Softmax, self).__init__()
        self.temp = config.temp
        
    def forward(self, x, logits, labels=None):
        """
        模型的前向传播函数，计算输出概率分布和损失。

        参数:
        x (Tensor): 输入数据,特征。
        logits (Tensor): 模型输出概率（未经过 softmax 的结果）。
        labels (Tensor, optional): 真实标签。默认为 None。

        返回值:
        tuple: 包含 softmax 后的概率分布和损失值的元组。
        """
        logits = F.softmax(logits)  # 计算模型输出的softmax概率分布
        if labels is None:  # 如果没有提供标签
            return logits, 0  # 直接返回softmax后的概率分布和0损失
        # 计算交叉熵损失，其中self.temp是温度参数，用于调整softmax输出的分布
        # 温度参数可以改变分布的熵，较低的温度会使分布更尖锐，较高的温度会使分布更平滑
        loss = F.cross_entropy(logits / self.temp, labels)
        return logits, loss  # 返回softmax后的概率分布和计算得到的损失值