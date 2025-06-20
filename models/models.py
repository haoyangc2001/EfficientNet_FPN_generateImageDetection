import torch
import torch.nn as nn
import torch_dct as dct
from torchvision.models import efficientnet_b0
# from modelscope import EfficientNetImageProcessor, EfficientNetForImageClassification
# from efficientnet_pytorch import EfficientNet


def get_input_data(input_img, data='dct'):
    if data == 'dct':
        return dct.dct_2d(input_img)    # 对输入图像进行二维离散余弦变换，并返回变换后的结果。
    elif data == 'img':
        return input_img               # 直接返回输入图像




# #简单卷积神经网络方案
# class vgg_layer(nn.Module):
#     def __init__(self, nin, nout):
#         super(vgg_layer, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(nin, nout, 3, 1, 1),
#             nn.BatchNorm2d(nout),
#             nn.LeakyReLU(0.2)
#         )

#     def forward(self, input):
#         return self.main(input)

# class dcgan_conv(nn.Module):
#     def __init__(self, nin, nout):
#         super(dcgan_conv, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(nin, nout, 4, 2, 1),
#             nn.BatchNorm2d(nout),
#             nn.LeakyReLU(0.2),
#         )

#     def forward(self, input):
#         return self.main(input)

# class Simple_CNN(nn.Module):
#     """
#     一个简单的卷积神经网络(CNN)模型,用于图像分类任务。

#     Args:
#         class_num (int): 分类类别数
#         out_feature_result (bool): 是否同时输出特征和分类结果,默认为False

#     Attributes:
#         main (nn.Sequential): 卷积层和vgg层组成的特征提取部分
#         fc (nn.Linear): 全连接层，用于特征降维
#         classification_head (nn.Sequential): 分类头部，用于图像分类
#         out_feature_result (bool): 是否同时输出特征和分类结果
#     """

#     def __init__(self, class_num, out_feature_result=False):
#         """
#         初始化Simple_CNN模型。

#         Args:
#             class_num (int): 分类类别数
#             out_feature_result (bool): 是否同时输出特征和分类结果,默认为False
#         """
#         super(Simple_CNN, self).__init__()
#         nf = 64 # 基础通道数
#         nc = 3  # 输入图像通道数（RGB图像为3）

#         # 特征提取部分，由卷积层和vgg层组成
#         self.main = nn.Sequential(
#             dcgan_conv(nc, nf),  # 第一个卷积块，输入通道数为nc，输出通道数为nf
#             vgg_layer(nf, nf),  # 第一个vgg层，输入通道数和输出通道数均为nf

#             dcgan_conv(nf, nf * 2),  # 第二个卷积块，输入通道数为nf，输出通道数为nf*2
#             vgg_layer(nf * 2, nf * 2),  # 第二个vgg层，输入通道数和输出通道数均为nf*2

#             dcgan_conv(nf * 2, nf * 4),  # 第三个卷积块，输入通道数为nf*2，输出通道数为nf*4
#             vgg_layer(nf * 4, nf * 4),  # 第三个vgg层，输入通道数和输出通道数均为nf*4

#             dcgan_conv(nf * 4, nf * 8),  # 第四个卷积块，输入通道数为nf*4，输出通道数为nf*8
#             vgg_layer(nf * 8, nf * 8),  # 第四个vgg层，输入通道数和输出通道数均为nf*8
#         )

#         # 全连接层，用于特征降维
#         self.fc = nn.Linear(nf * 8 * 8 * 8, nf * 8, bias=True)

#         # 分类头部，用于图像分类
#         self.classification_head = nn.Sequential(
#             nn.Dropout(p=0.2, inplace=True),  # Dropout层，防止过拟合
#             nn.Linear(nf * 8, class_num, bias=True)  # 全连接层，输出类别数
#         )
#         # 是否同时输出特征和分类结果
#         self.out_feature_result = out_feature_result


#     def forward(self, input, data='dct'):   # input:[88,3,128,128]
#         input = get_input_data(input, data) # input:[88,3,128,128]
#         embedding = self.main(input)        # embedding:[88,512,8,8]
#         feature = embedding.view(embedding.shape[0], -1) #  feature:[88,512*8*8=32768]
#         feature = self.fc(feature)                       # feature:[88,512]
#         cls_output = self.classification_head(feature)   # cls_output:[88,11]

#         if self.out_feature_result:
#             return cls_output, feature
#         else:
#             return cls_output







# # 预训练 efficientNet 方案
# class Simple_CNN(nn.Module):
#     def __init__(self, class_num, out_feature_result=False):
#         super(Simple_CNN, self).__init__()
        
#         self.model = efficientnet_b0(weights=None)
        

#             # 加载预训练权重并映射到CPU
#         weights_path = '/mnt/workspace/POSE/dataset/pretrained_models/EfficientNet/efficientnet_b0_rwightman-7f5810bc.pth'
#         weights = torch.load(weights_path, map_location=torch.device('cpu'))
#         self.model.load_state_dict(weights)
        
#         # 是否部分微调
#         # for param in self.model.parameters():
#         #     param.requires_grad = False

#         # # Unfreeze last few layers
#         # for name, param in self.model.named_parameters():
#         #     if 'blocks.6' in name or 'conv_head' in name:
#         #         param.requires_grad = True

#         # Modify the output layer for n classification
#         linear_layer = self.model.classifier[1]
#         num_ftrs = linear_layer.in_features
#         self.model.classifier[1] = nn.Linear(num_ftrs, class_num)

#         # 是否同时输出特征和分类结果
#         self.out_feature_result = out_feature_result


#     def forward(self, input, data='dct'):          # input:[88,3,128,128]
#         processed_input = get_input_data(input, data)   # processed_input [88,3,128,128]    

#         # 获取特征提取器和分类器
#         features = self.model.features(processed_input)   # features [88,512,4,4]

#         # 应用全局平均池化
#         x = nn.functional.adaptive_avg_pool2d(features, (1, 1))  # x [88,512,1,1]
#         features = torch.flatten(x, 1)           # x [88,512]

#         # 通过分类器
#         cls_output = self.model.classifier(features)

#         # 根据out_feature_result决定返回什么
#         if self.out_feature_result:
#             # 返回分类结果和特征
#             return cls_output, features
#         else:
#             # 只返回分类结果
#             return cls_output
    




# 预训练EfficientNet_B0_FPN方案
class Simple_CNN(nn.Module):
    def __init__(self, class_num, out_feature_result=False):
        super(Simple_CNN, self).__init__()

        # 加载预训练权重并映射到CPU
        self.model = efficientnet_b0(weights=None)
        weights_path = '/mnt/workspace/POSE/dataset/pretrained_models/EfficientNet/efficientnet_b0_rwightman-7f5810bc.pth'
        weights = torch.load(weights_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(weights)
        


        #是否部分微调
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last few layers
        for name, param in self.model.named_parameters():
            if 'blocks.6' in name or 'conv_head' in name:
                param.requires_grad = True



        # 提取多尺度特征的层
        # EfficientNet-B0的通道数: [16, 24, 40, 112, 320]
        self.feature_blocks = nn.ModuleList([
            nn.Sequential(self.model.features[0], self.model.features[1]),  # Stage 1: 1/2, 16通道
            self.model.features[2],  # Stage 2: 1/4, 24通道
            self.model.features[3],  # Stage 3: 1/8, 40通道
            self.model.features[4],  # Stage 4: 1/16, 112通道
            self.model.features[5],  # Stage 5: 1/32, 320通道
        ])
        
        # 确定每个阶段的通道数
        # 使用dummy input来推断每个阶段的输出通道数
        dummy_input = torch.randn(1, 3, 128, 128)
        dummy_features = []
        x = dummy_input
        for block in self.feature_blocks:
            x = block(x)
            dummy_features.append(x)
        
        # 获取每个阶段的通道数
        in_channels = [f.shape[1] for f in dummy_features]
        # print(f"各阶段通道数: {in_channels}")
        
        # FPN参数 - 使用动态确定的通道数
        # 统一所有特征的通道数为128
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, 128, kernel_size=1) for channels in in_channels
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(128, 128, kernel_size=3, padding=1) for _ in in_channels
        ])
        
        # 最终分类器
        # 修改线性层输入维度以匹配特征融合后的维度 (128 * 5 = 640)
        self.classifier = nn.Sequential(
            nn.Linear(128 * len(in_channels), 512),  # 直接从特征融合后的维度开始
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, class_num)
        )

        # 是否同时输出特征和分类结果
        self.out_feature_result = out_feature_result

    def forward(self, input, data='dct'):          # input:[88,3,128,128]
        processed_input = get_input_data(input, data)   # processed_input [88,3,128,128]
        
        # 提取多尺度特征
        features = []
        x = processed_input
        for i, block in enumerate(self.feature_blocks):
            x = block(x)
            features.append(x)
        
        # FPN上采样与融合
        fpn_features = []
        # 从最深层开始
        fpn_features.append(self.fpn_convs[-1](self.lateral_convs[-1](features[-1])))
        
        # 上采样并融合浅层特征
        for i in range(len(features)-2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            # 上采样
            upsample = nn.functional.interpolate(
                fpn_features[0], 
                size=lateral.shape[2:], 
                mode='nearest'
            )
            # 融合
            fused = lateral + upsample
            fpn_features.insert(0, self.fpn_convs[i](fused))
        
        # 特征融合（这里使用简单的平均池化）
        # 对每个FPN特征进行池化并拼接
        pooled_features = []
        for feat in fpn_features:
            pooled = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            pooled_features.append(pooled)
        
        # 合并多尺度特征
        fused_features = torch.cat(pooled_features, dim=1)  # [B, 128*5, 1, 1]
        
        # 应用全局平均池化并展平
        x = nn.functional.adaptive_avg_pool2d(fused_features, (1, 1))  # [B, 128*5, 1, 1]
        x = torch.flatten(x, 1)  # [B, 128*5]
        
        # 通过分类器
        features = self.classifier[0](x)  # [B, 512]
        cls_output = self.classifier[1:](features)  # [B, class_num]

        # 根据out_feature_result决定返回什么
        if self.out_feature_result:
            # 返回分类结果和特征
            return cls_output, features
        else:
            # 只返回分类结果
            return cls_output    