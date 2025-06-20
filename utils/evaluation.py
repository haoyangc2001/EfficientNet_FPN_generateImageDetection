import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score


def evaluate_multiclass(gt_labels, pred_labels):
    """
    评估多分类模型的性能。

    参数:
        gt_labels (array-like): 真实标签，表示样本的实际类别。
        pred_labels (array-like): 预测标签，表示模型对样本类别的预测结果。

    返回值:
        dict: 包含多个性能指标的字典，包括每类的召回率、宏平均召回率、宏平均 F1 分数和准确率。
    """
    # 计算准确率：分类正确的样本数占总样本数的比例
    acc = accuracy_score(gt_labels, pred_labels)
    
    # 计算宏平均 F1 分数：每个类别的 F1 分数的简单平均，不考虑类别不平衡
    f1 = f1_score(gt_labels, pred_labels, average='macro')
    
    # 计算宏平均召回率：每个类别的召回率的简单平均
    recall = recall_score(gt_labels, pred_labels, average='macro')
    
    # 计算每类的召回率，返回一个数组，其中每个元素对应一个类别的召回率
    recalls = recall_score(gt_labels, pred_labels, average=None)

    # 返回包含多个性能指标的字典
    return {'recalls': recalls, 'recall': recall, 'f1': f1, 'acc': acc}

def get_curve_online(known, novel, stypes = ['Bas']):
    '''计算不同评估指标所需的曲线数据，包括真阳性率（TPR）、假阳性率（FPR）和在 TPR 为 95% 时的真负率（TNR）。'''
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(x1, x2, stypes = ['Bas'], verbose=True):
    '''通过计算多种指标（如 TNR、AUROC 等），全面评估模型在检测异常数据（未知类别）方面的性能。'''
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        roc_auc = 100.*(-np.trapz(1.-fpr, tpr))
        results[stype][mtype] = roc_auc
        results[stype]['tpr']=tpr
        results[stype]['fpr']=fpr

        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            print('')
    
    return results    #一个字典，包含每种统计类型对应的评估指标结果

def compute_oscr(pred_k, pred_u, labels):
    """
    计算开放集识别（OSCR）分数。

    OSCR 分数衡量模型在开放集识别任务中的性能，综合考虑了模型对已知类别的正确分类能力和对未知类别的检测能力。

    参数:
    pred_k (numpy.ndarray): 已知类别数据的预测概率矩阵，形状为 (num_samples, num_classes)。
    pred_u (numpy.ndarray): 未知类别数据的预测概率矩阵，形状为 (num_samples, num_classes)。
    labels (numpy.ndarray): 已知类别数据的真实标签，形状为 (num_samples,)。

    返回值:
    float: OSCR 分数，表示模型在开放集识别任务中的性能。
    """

    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)  # pred_k和pred_u预测概率的最大值
    pred = np.argmax(pred_k, axis=1)  # known样本的预测值
    correct = (pred == labels)  # 判断预测是否正确
    m_x1 = np.zeros(len(x1))  # 初始化已知样本的正确分类标记
    m_x1[pred == labels] = 1  # 正确分类的样本标记为1

    # 创建目标标签：已知样本的正确分类标记和未知样本的标记
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)  # 已知样本的目标标签（正确分类为1，错误分类为0）
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)  # 未知样本的目标标签（1表示未知）

    # 合并已知和未知样本的最大预测概率
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)  # 总样本数

    # 初始化正确分类率（CCR）和假阳性率（FPR）
    CCR = [0 for _ in range(n + 2)]
    FPR = [0 for _ in range(n + 2)]

    # 按预测概率从小到大排序
    idx = predict.argsort()
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    # 计算每个阈值下的 CCR 和 FPR
    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()  # 正确分类的样本数
        FP = s_u_target[k:].sum()  # 假阳性的样本数

        CCR[k] = float(CC) / float(len(x1))  # 正确分类率
        FPR[k] = float(FP) / float(len(x2))  # 假阳性率

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # 组合 FPR 和 CCR 为 ROC 曲线上的点
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # 使用梯形法则计算 ROC 曲线下的面积（OSCR）
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0
        OSCR = OSCR + h * w

    return OSCR


def metric_cluster(X_selected, n_clusters, y, cluster_method='kmeans'):
    """
    计算聚类结果的评估指标：归一化互信息（NMI）、纯度（Purity）和调整兰德指数（ARI）。

    参数:
        X_selected: {numpy array}, shape (n_samples, n_selected_features)
            输入数据，包含样本在选定特征上的值。
        n_clusters: {int}
            聚类的数目。
        y: {numpy array}, shape (n_samples,)
            真实标签。
        cluster_method: {str}, optional, default='kmeans'
            聚类方法，可选 'kmeans', 'minibatch_kmeans', 'dbscan'。

    输出:
        nmi: {float}
            归一化互信息，衡量聚类结果与真实标签之间的一致性。
        purity: {float}
            纯度，衡量聚类结果的 purity。
        ari: {float}
            调整兰德指数，衡量聚类结果与真实标签之间的相似性。

    注意:
        - 当选择 'dbscan' 作为聚类方法时，需注意其对参数（如 eps 和 min_samples）的敏感性。
        - 不同的聚类方法可能对数据的预处理有不同的要求。
    """
    # 根据指定的聚类方法初始化聚类算法
    if cluster_method == 'kmeans':
        cluster_alg = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001, precompute_distances=True, verbose=0,
                     random_state=None, copy_x=True, n_jobs=1)
    elif cluster_method == 'minibatch_kmeans':
        cluster_alg = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, batch_size=2048)
    elif cluster_method == 'dbscan':
        cluster_alg = DBSCAN(eps=3, min_samples=2)
    else:
        raise ValueError('select kmeans or dbscan for cluster')

    # 对输入数据进行聚类
    cluster_alg.fit(X_selected)
    y_predict = cluster_alg.labels_

    # 计算聚类评估指标
    nmi, purity, ari = cluster_stats(y_predict, y)

    return nmi, purity, ari



def cluster_stats(predicted, targets, save_path=None):
    n_clusters = np.unique(predicted).size
    n_classes  = np.unique(targets).size
    num = np.zeros([n_clusters,n_classes])
    unique_targets = np.unique(targets)
    for i,p in enumerate(np.unique(predicted)):
        class_labels = targets[predicted==p]
        num[i,:] = np.sum(class_labels[:,np.newaxis]==unique_targets[np.newaxis,:],axis=0)
    sum_clusters = np.sum(num,axis=1)
    purity = np.max(num,axis=1)/(sum_clusters+(sum_clusters==0).astype(sum_clusters.dtype))
    indices = np.argsort(-purity)

    if save_path is not None:
        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(purity[indices],color='red')
        ax1.set_xlabel('Cluster index')
        ax1.set_ylabel('Purity')
        ax2 = ax1.twinx()
        ax2.plot(sum_clusters[indices])
        ax2.set_ylabel('Cluster size')
        plt.legend(('Purity','Cluster size'))
        plt.show()
        plt.title('Cluster size and purity of discovered clusters')
        plt.savefig(save_path)
    print('Data points {} Clusters {}'.format(np.sum(sum_clusters).astype(np.int64), n_clusters))
    print('Average purity: {:.4f} '.format(np.sum(purity*sum_clusters)/np.sum(sum_clusters))+\
          'NMI: {:.4f} '.format(normalized_mutual_info_score(targets, predicted))+\
          'ARI: {:.4f} '.format(adjusted_rand_score(targets, predicted)))
    avg_purity = np.sum(purity*sum_clusters)/np.sum(sum_clusters) 
    nmi = normalized_mutual_info_score(targets, predicted) 
    ari = adjusted_rand_score(targets, predicted) 
    return nmi, avg_purity, ari
