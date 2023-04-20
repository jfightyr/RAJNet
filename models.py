from functools import partial#model找问题，要经常打断点，找到变化比较异常的点进行模型改进#画出来模型的连接图
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_inplanes():#输入通道
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class ChannelAttention(nn.Module):
    def __init__(self,inplanes,ratio=16):
        super(ChannelAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(inplanes,inplanes//16,1,bias=False) # 本来是2的改成3了
        self.relu1=nn.ReLU()
        self.fc2=nn.Conv3d(inplanes//16,inplanes,1,bias=False)

        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        avg_out = self.avg_pool(x)
        
        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:#downsample??
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from swin_transformer import SwinTransformer
# from model.swin_transformer import SwinTransformer

# 定义动作识别模型
class ActionRecognitionModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.patch_size = 4 # Swin-B 的 patch size
        self.hidden_dim = 96 # Swin-B 的隐藏维度
        self.num_heads = 8 # Swin-B 的头数
        self.num_layers = 2 # Swin-B 的层数
        self.expand_ratio = 4 # Swin-B 的扩张比例
        self.dropout_rate = 0.5 # Swin-B 的 dropout rate
        self.embed_dim = self.hidden_dim * self.patch_size ** 2 // self.expand_ratio

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(SwinTransformer, self.num_layers, self.embed_dim, self.hidden_dim, self.num_heads, self.expand_ratio, self.dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(self.hidden_dim * 2, num_classes)

    def _make_layer(self, block, num_layers, embed_dim, hidden_dim, num_heads, expand_ratio, dropout_rate):
        layers = []
        for i in range(num_layers):
            layers.append(block(embed_dim if i == 0 else hidden_dim, hidden_dim, num_heads, expand_ratio, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # 将特征向量和前两个时间步的平均值拼接
        x = torch.cat([x, x[:, :2].mean(dim=1)], dim=1)
        x = self.fc(x)
        return x


# 扩散模型 但是不太对？
# class ActionRecognition(nn.Module):  
#     def __init__(self, num_classes):
#         super(ActionRecognition, self).__init__()
        
#         # 加载预训练的扩散模型
#         self.diffusion_model = torch.load('diffusion_model.pth')
        
#         # 获取扩散模型的输出大小
#         with torch.no_grad():
#             sample_data = torch.randn(1, 3, 16, 224, 224)
#             diffused_data = self.diffusion_model(sample_data)
#             _, _, c, t, h, w = diffused_data.size()
        
#         # 修改全连接层的输入大小
#         self.fc = nn.Linear(c*t*h*w, num_classes)
        
#         # 冻结扩散模型的参数
#         for param in self.diffusion_model.parameters():
#             param.requires_grad = False
    
#     def forward(self, x):
#         # 通过扩散模型获取特征
#         with torch.no_grad():
#             x = self.diffusion_model(x)
        
#         # 将特征展平
#         x = x.view(x.size(0), -1)
        
#         # 经过全连接层
#         x = self.fc(x)
        
#         return x

# ChatGPT 编写的提升性能模块 可以不加
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = nn.Sequential()
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(residual)
        out = self.relu(out)

        return out

# chatGPT 预训练模块
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from pytorch_i3d import InceptionI3d

# class I3DFeatureExtractor(nn.Module):
#     def __init__(self, num_classes):
#         super(I3DFeatureExtractor, self).__init__()
#         self.i3d = InceptionI3d(num_classes=num_classes, in_channels=3)

#     def forward(self, x):
#         features = self.i3d.extract_features(x)
#         return features

#https://blog.csdn.net/weixin_55073640/article/details/122552814讲的不错
class ResNet(nn.Module):  # 核心 要画出怎么连的

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,  # 块数
                 n_input_channels=3,  # 输入RGB三个通道 
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()# 构造父类

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]# <list>类型填充 why?

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        # self.pretrained = I3DFeatureExtractor()
        # self.chat_res = ResNetBlock() # 增强性能模块，可以不加
        self.conv1 = nn.Conv3d(n_input_channels,  # 通道数
                               self.in_planes,  # 通道数
                               kernel_size=(conv1_t_size, 7, 7),  #卷积核
                               stride=(conv1_t_stride, 2, 2),  # 步长
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        # self.ActionRecognition=ActionRecognition(2)
         #在卷积层最后一层加入注意力机制
        
        self.ca1=ChannelAttention(self.in_planes)
        self.sa1 = SpatialAttention()
        # self.swinB=ActionRecognitionModel()

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  #三维的所以有三个参数
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)  #全连接层
        self.pro = nn.Softmax(dim = 1) 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # 最重要的 对比RESNET经典网络结构去看 (1,1,1,256,256)

        x = self.conv1(x) # BUG 将numpy转化为float32，解决volume.astype(np.float32) (1,1,1,256,256)
        x = self.bn1(x)
        x = self.relu(x)

        # === 加入预训练模块
        # x = self.pretrained(x)
        # === 
        
        # === 插入预训练模型 chatgpt编写一个
        # x = self.chat_res(x) # 效用不大，可以不加
        # === 

        if not self.no_max_pool:
            x = self.maxpool(x)
        
        # x = self.swinB(x,2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.ca1(x)  
        # x = self.sa1(x)
        x = self.avgpool(x)  #平均池化

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.pro(x) # 归一化
        
        return x


def generate_model(model_depth, **kwargs):#向函数传递 可变参数
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

if __name__ == '__main__':
    model = generate_model(model_depth=10, n_input_channels=1, n_classes=2)
    print(model)

    x = torch.randn(4, 1, 64, 224, 224)
    res = model(x)
    print(res.shape)


