import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        计算度量学习损失。

        参数:
            inputs: 特征矩阵，形状为 (batch_size, feat_dim)
            targets: 真实标签，形状为 (num_classes)

        返回值:
            loss: 计算得到的损失值
        """
        n = inputs.size(0)  # 获取批次大小

        # inputs = 1.0 * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # 对输入特征进行归一化（可选）

        # 计算成对距离
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # 计算每个样本的平方欧氏范数
        dist = dist + dist.t()  # 矩阵转置相加
        dist.addmm_(1, -2, inputs, inputs.t())  # 计算成对距离矩阵，得到每个样本之间的欧氏距离的平方
        dist = dist.clamp(min=1e-12).sqrt()  # 保证数值稳定性并开平方

        # 对于每个锚点，找到最难的正样本和负样本
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())  # 创建掩码 n×n 的矩阵，比较两个矩阵中对应位置的标签，相同标签的位置为 True
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 从特征距离矩阵 dist 中提取出样本 i 到所有正样本（即标签相同的样本）的距离。选出最大值，表示最难的正样本距离。
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 从特征距离矩阵 dist 中提取出样本 i 到所有负样本（即标签不同的样本）的距离。选出最小值，表示最难的负样本距离。
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # 计算排序 hinge 损失
        y = torch.ones_like(dist_an)  # 创建与 dist_an 形状相同的全 1 张量
        # 对于同一类别的样本（正样本），它们的特征表示应该更接近。
        # 对于不同类别的样本（负样本），它们的特征表示应该更远离，并且至少保持一个边距的距离
        loss = self.ranking_loss(dist_an, dist_ap, y)  # 使用 hinge 损失计算排序损失，loss=max(0,margin−(dist_an−dist_ap))


        if self.mutual:
            return loss, dist  # 如果启用了相互学习，返回损失和距离矩阵

        return loss  # 返回损失
