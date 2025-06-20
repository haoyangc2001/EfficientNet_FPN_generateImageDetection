import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.common import read_annotations
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, annotations, config, balance=False, test_mode=False):
        """
        初始化数据集对象

        Args:
        annotations (list): 原始标注数据列表，格式为 [(图像路径, 标签), ...]
        config (object): 配置对象，需包含以下属性：
        - resize_size (int/tuple): 图像缩放尺寸
        - class_num (int): 数据集的类别总数
        balance (bool, optional): 是否按类别平衡数据. 默认为False
        test_mode (bool, optional): 是否为测试模式. 默认为False
        """
        # 从配置对象获取参数
        self.resize_size = config.resize_size # 图像预处理缩放尺寸
        self.class_num = config.class_num     # 数据集类别总数
        self.balance = balance                # 是否启用类别平衡采样
        self.test_mode = test_mode            # 测试模式标识（可能影响数据增强等行为）
        self.config = config                  # 存储完整配置对象

        # 定义标准化预处理流程
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),                  # 将PIL/numpy图像转为Tensor
            transforms.Normalize(                   # 标准化处理（将[0,1]映射到[-1,1]范围）
                mean=[0.5, 0.5, 0.5],               # RGB三通道均值（0.5对应输入范围为[0,1]的情况）
                std=[0.5, 0.5, 0.5]                 # RGB三通道标准差
            )
        ])

        # 数据重组逻辑
        if balance:
            # 按类别分组数据：生成嵌套列表结构，外层为类别，内层为对应样本
            # 示例：[[类别0样本...], [类别1样本...], ...]
            self.data = [
                [x for x in annotations if x[1] == lab]  # 筛选当前类别样本（假设标签在x[1]位置）
                for lab in range(config.class_num)       # 遍历所有类别（0到class_num-1）
            ]
        else:
            # 不启用平衡时，保持原始数据列表结构
            self.data = [annotations]  # 嵌套列表结构保持统一（外层为单元素列表）   
   
    def __len__(self):
        return max([len(subset) for subset in self.data])

    def __getitem__(self, index):
        """
        根据索引获取数据样本（支持类别平衡模式）
        
        Args:
            index (int): 数据索引(当balance=True时,实际会进行跨类别采样)
            
        Returns:
            - balance=True时:
                images (Tensor): 维度为 [class_num, channels, H, W] 的张量，每个类别各取一个样本
                labels (LongTensor): 长度为 class_num 的标签张量
                img_paths (list): 对应各样本的原始文件路径列表
            
            - balance=False时:
                image (Tensor): 单个图像张量 [channels, H, W]
                label (LongTensor): 单个标签值
                img_path (str): 单个文件路径
        """

        
        if self.balance:
            # 类别平衡模式：从每个类别中各取一个样本组成batch
            labs = []        # 存储所有类别的标签
            imgs = []        # 存储所有类别的图像数据
            img_paths = []   # 存储所有类别的文件路径
            
            # 遍历每个类别
            for i in range(self.class_num):
                # 安全索引计算（循环使用当前类别的样本）
                safe_idx = index % len(self.data[i])  # 防止索引越界，当index超过类别样本数时循环使用
                # 获取样本信息
                img_path, lab = self.data[i][safe_idx]  # 假设self.data是嵌套列表结构[[类别0样本], [类别1样本],...]
                img = self.load_sample(img_path)        # 加载并预处理图像（假设包含resize/transform等操作）
                # 收集数据
                labs.append(lab)
                imgs.append(img)
                img_paths.append(img_path)
        
            # 将图像列表转换为形状为 [class_num, C, H, W] 的张量
            # unsqueeze(0)为添加批次维度，cat合并后得到多类别样本组成的伪batch
            return (
                torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]),  # 拼接成维度[class_num, C, H, W]
                torch.tensor(labs, dtype=torch.long),  # 转换为LongTensor标签
                img_paths  # 返回所有样本路径
            )
        else:
            # 普通模式：直接返回单个样本
            img_path, lab = self.data[0][index]  # 当balance=False时，self.data结构为[全部样本]
            img = self.load_sample(img_path)    # 加载图像
            lab = torch.tensor(lab, dtype=torch.long)  # 转换为张量
            
            return img, lab, img_path  # 返回单样本数据


    def load_sample(self, img_path):
        """
        加载并处理图像样本。

        Args:
            img_path (str): 图像文件路径

        Returns:
            Tensor: 处理后的图像张量
        """
        # 打开图像文件并转换为RGB模式
        img = Image.open(img_path).convert('RGB')
        
        # 如果图像不是正方形，将其裁剪为正方形
        if img.size[0] != img.size[1]:
            # 获取图像的较短边作为裁剪大小
            short_size = img.size[0] if img.size[0] < img.size[1] else img.size[1]
            # 使用中心裁剪将图像裁剪为正方形
            img = transforms.CenterCrop(size=(short_size, short_size))(img)

        # 如果设置了目标大小，将图像调整大小
        if self.resize_size is not None:
            img = img.resize(self.resize_size)
        
        # 应用归一化转换
        img = self.norm_transform(img)

        return img

class BaseData(object):
    def __init__(self, script_dir, train_data_path, val_data_path,
                test_data_path, out_data_path, 
                opt, config):
        """
        基础数据加载类，用于初始化各类数据集加载器
        
        Args:
            script_dir(str): 当前的脚本目录
            train_data_path (str): 训练集标注文件路径
            val_data_path (str): 验证集标注文件路径
            test_data_path (str): 测试集标注文件路径
            out_data_path (str): 外部数据集标注文件基础路径（需支持字符串替换）
            opt (object): 命令行参数对象,包含debug等模式参数
            config (object): 配置对象，包含以下关键参数：
                - num_workers (int): 数据加载并行进程数
                - batch_size (int): 批次大小
                - 其他模型相关配置参数
        """
        # 初始化训练数据集和数据加载器
        train_set = ImageDataset(
            read_annotations(script_dir, train_data_path, opt.debug),  # 读取训练集标注文件 [(样本1路径,标签),(样本2路径,标签)……]
            config, 
            balance=True  # 启用类别平衡采样
        )
        # 训练数据加载器配置（打乱顺序、保留不完整批次）
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,  # 设置为CPU核心数的8倍
            batch_size=config.batch_size,    # 批量大小 8
            pin_memory=True,                 # 加速GPU数据传输
            shuffle=True,                    # 训练数据需要打乱
            drop_last=False,                 # 保留最后不完整批次
        )

        # # 验证集配置（关闭平衡采样，启用测试模式）
        # val_set = ImageDataset(
        #     read_annotations(script_dir, val_data_path, opt.debug),
        #     config,
        #     balance=False,    # 验证集无需平衡采样
        #     test_mode=True    # 关闭数据增强等训练模式特有操作
        # )
        # val_loader = DataLoader(
        #     dataset=val_set,
        #     num_workers=config.num_workers,
        #     batch_size=config.batch_size,
        #     pin_memory=True,
        #     shuffle=True,    # 验证集可保持打乱（或根据需求关闭）
        #     drop_last=False,
        # )

        # TSNE可视化专用加载器（平衡采样，丢弃末尾批次）
        tsne_set = ImageDataset(
            read_annotations(script_dir, test_data_path, opt.debug),
            config,
            balance=True,     # 平衡采样便于可视化各类别
            test_mode=True
        )
        tsne_loader = DataLoader(
            dataset=tsne_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,    # 确保批次完整，避免可视化异常
        )

        # 标准测试集配置
        test_set = ImageDataset(
            read_annotations(script_dir, test_data_path, opt.debug),
            config,
            balance=False,
            test_mode=True
        )
        test_loader = DataLoader(
            dataset=test_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,      # 测试集可根据需求关闭打乱
            drop_last=False,
        )

        # 外部基础数据集
        out_set = ImageDataset(
            read_annotations(script_dir, out_data_path, opt.debug),
            config,
            balance=False,
            test_mode=True
        )
        out_loader = DataLoader(
            dataset=out_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False
        )

        # # 外部数据变种1：种子相关数据（路径替换逻辑）
        # out_set1 = ImageDataset(
        #     read_annotations(script_dir, out_data_path.replace('out','out_seed'), opt.debug),  # 路径替换生成新数据源
        #     config,
        #     balance=False,
        #     test_mode=True
        # )
        # out_loader1 = DataLoader(
        #     dataset=out_set1,
        #     num_workers=config.num_workers,
        #     batch_size=config.batch_size,
        #     pin_memory=True,
        #     shuffle=True,
        #     drop_last=False
        # )

        # # 外部数据变种2：架构相关数据
        # out_set2 = ImageDataset(
        #     read_annotations(script_dir, out_data_path.replace('out','out_arch'), opt.debug),
        #     config,
        #     balance=False,
        #     test_mode=True
        # )
        # out_loader2 = DataLoader(
        #     dataset=out_set2,
        #     num_workers=config.num_workers,
        #     batch_size=config.batch_size,
        #     pin_memory=True,
        #     shuffle=True,
        #     drop_last=False
        # )

        # # 外部数据变种3：数据集相关数据
        # out_set3 = ImageDataset(
        #     read_annotations(script_dir, out_data_path.replace('out','out_dataset'), opt.debug),
        #     config,
        #     balance=False,
        #     test_mode=True
        # )
        # out_loader3 = DataLoader(
        #     dataset=out_set3,
        #     num_workers=config.num_workers,
        #     batch_size=config.batch_size,
        #     pin_memory=True,
        #     shuffle=True,
        #     drop_last=False
        # )

        # # 将加载器绑定到实例属性
        # self.out_loader1 = out_loader1  # 种子变种加载器
        # self.out_loader2 = out_loader2  # 架构变种加载器
        # self.out_loader3 = out_loader3  # 数据集变种加载器

        self.train_loader = train_loader  # 主训练数据流
        # self.val_loader = val_loader      # 验证集数据流
        self.test_loader = test_loader    # 测试集数据流
        self.out_loader = out_loader      # 基础外部数据流
        self.tsne_loader = tsne_loader    # 可视化专用数据流

        # 打印各数据集样本量统计（用于数据校验）
        # print('train: {}, val: {}, test {}, out {}'.format(len(train_set), len(val_set), len(test_set), len(out_set)))
        print('train: {}, test {}, out {}'.format(len(train_set), len(test_set), len(out_set)))
