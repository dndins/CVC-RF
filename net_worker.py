import torch
from torch import nn
from torchvision.models import resnet18
import torch.nn.functional as F
from mvit.mvit.models.attention import MultiScaleBlock

class BasicBlock(nn.Module):
    def __init__(self,inplanes: int,planes: int,stride: int = 1,downsample = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):

    def __init__(self, layers=[2, 2, 2, 2], num_classes=128, zero_init_residual=False):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.do = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

        self.layer1_channel_trans = nn.Sequential(nn.Conv2d(64, 128, 1, 1, 1),
                                                  nn.AdaptiveAvgPool2d((7, 7)))
        self.layer2_channel_trans = nn.Sequential(nn.Conv2d(128, 128, 1, 1, 1),
                                                  nn.AdaptiveAvgPool2d((7, 7)))
        self.layer3_channel_trans = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 1),
                                                  nn.AdaptiveAvgPool2d((7, 7)))
        self.layer4_channel_trans = nn.Sequential(nn.Conv2d(512, 128, 1, 1, 1),
                                                  nn.AdaptiveAvgPool2d((7, 7)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, planes: int, blocks: int,stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes))
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        #先做7x7的卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1_out = self.layer1(x)
        # x1 = self.layer1_channel_trans(x1_out)
        x2_out = self.layer2(x1_out)
        # x2 = self.layer2_channel_trans(x2_out)
        x3_out = self.layer3(x2_out)
        # x3 = self.layer3_channel_trans(x3_out)
        x4_out = self.layer4(x3_out)
        # x4 = self.layer4_channel_trans(x4_out)

        x = self.avgpool(x4_out)
        x = torch.flatten(x, 1)
        #x = self.do(x)
        x = self.fc(x)
        
        return [x1_out, x2_out, x3_out, x4_out], x

def get_backbone():
    weight = torch.load(r'/mnt/data1/zzy/ProjecT/B_ProJ_bank_mvit/resnet18-5c106cde.pth')
    net = ResNet18(num_classes=1000)
    model_dict = net.state_dict()
    new_state_dict = {k: v for k, v in weight.items() if k in model_dict}
    weight.update(new_state_dict)
    net.load_state_dict(weight, strict=False)

    input_num = net.fc.in_features
    net.fc = nn.Sequential(nn.Linear(input_num, 128))

    for k in weight.keys():
        if k in weight:
            print(f"{k}: 加载成功")
        else:
            print(f"{k}: 使用默认初始化")
    
    return net

class simple_classifiar(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(simple_classifiar, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim//2)
        self.fc2 = nn.Linear(in_dim//2, out_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

class NetWork(nn.Module):
    def __init__(self, num_class):
        super(NetWork, self).__init__()
        self.backbone = get_backbone()
        self.spacing_mlp = nn.Sequential(  
                            nn.Linear(1, 256),
                            nn.ReLU(),
                            nn.Linear(256, 512)
                        )
        self.exp1 = nn.Sequential(nn.Linear(1024, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 256))
        
        self.exp2 = nn.Sequential(nn.Linear(1024, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 256))
        
        # self.exp3 = nn.Sequential(nn.Linear(1024, 512),
        #                           nn.ReLU(),
        #                           nn.Linear(512, 256))
        
        # self.exp4 = nn.Sequential(nn.Linear(1024, 512),
        #                           nn.ReLU(),
        #                           nn.Linear(512, 256))
        
        self.simple_class = simple_classifiar(256, num_class)
        self.mvit1 = MultiScaleBlock(dim=64, dim_out=128, input_size=(112, 112), kernel_q=(2, 2), kernel_kv=(2, 2), stride_q=(2, 2), stride_kv=(2, 2))
        self.mvit2 = MultiScaleBlock(dim=128, dim_out=256, input_size=(56, 56))
        self.mvit3 = MultiScaleBlock(dim=256, dim_out=512, input_size=(28, 28))
        self.mvit4 = MultiScaleBlock(dim=512, dim_out=512, input_size=(14, 14))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.gate = nn.Sequential(nn.Linear(1024, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 3))
        
    def forward(self, x, spacing, va, vb):
        x_out, x = self.backbone(x)
        
        B1, C1, H1, W1 = x_out[0].shape
        B2, C2, H2, W2 = x_out[1].shape
        B3, C3, H3, W3 = x_out[2].shape
        B4, C4, H4, W4 = x_out[3].shape
        x_layer1 = x_out[0].view(B1, C1, -1).permute(0, 2, 1) # [B, hw, C]
        x_layer1_out, x_layer1_shape = self.mvit1(x_layer1, hw_shape=(H1, W1))
       
        x_layer2 = x_out[1].view(B2, C2, -1).permute(0, 2, 1) # [B, hw, C]
        x_layer2 = x_layer1_out+ x_layer2
        x_layer2_out, x_layer2_shape = self.mvit2(x_layer2, hw_shape=(H2, W2))
        
        x_layer3 = x_out[2].view(B3, C3, -1).permute(0, 2, 1) # [B, hw, C]
        x_layer3 = x_layer2_out + x_layer3
        x_layer3_out, x_layer3_shape = self.mvit3(x_layer3, hw_shape=(H3, W3))

        x_layer4 = x_out[3].view(B4, C4, -1).permute(0, 2, 1) # [B, hw, C]
        x_layer4 = x_layer3_out + x_layer4
        x_layer4_out, x_layer3_shape = self.mvit4(x_layer4, hw_shape=(H4, W4))
        
        x_layer4_out = self.avgpool(x_layer4_out.permute(0, 2, 1)).squeeze(-1)
        # spacing = self.spacing_mlp(spacing.unsqueeze(1))     
        # spacing = torch.cat([spacing, spacing], dim=0)
        xa, xb = x[:x.size(0)//2], x[x.size(0)//2:] # 最后一层特征用来对比学习
        x_layer4_out = x_layer4_out
        x = torch.cat([x_layer4_out[:x_layer4_out.size(0)//2], x_layer4_out[x_layer4_out.size(0)//2:]], dim=-1)  



        # x_feat_fusion = self.atten(x_multi_layers_feat) + spacing

        center_v_A = va.unsqueeze(0).repeat(xa.size(0), 1, 1) # [32, 3, 128]
        center_v_B = vb.unsqueeze(0).repeat(xb.size(0), 1, 1) # [32, 3, 128]
        
        center = torch.cat([center_v_A, center_v_B], dim=-1)
        xab = torch.cat([xa, xb], dim=-1)


        weight = F.softmax(F.cosine_similarity(xab.unsqueeze(1), center, dim=-1), dim=-1) # [32, 3]
        # weight = F.softmax(self.gate(x), dim=-1)
        # print(weight[0])
        x = torch.einsum('b n d, b n -> b d', torch.stack([self.exp1(x), self.exp2(x)], dim=1), weight)

        x = self.simple_class(x)
        
        return xa, xb, x

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


# -------------------------------------VIT----------------------------------------------------
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ff_dim=128, dropout=0.1):
        """
        Transformer Layer
        :param embed_dim:  输入特征维度 (d_model)
        :param num_heads:  多头注意力的头数
        :param ff_dim:  前馈网络的隐藏层维度
        :param dropout:  Dropout 概率
        """
        super(TransformerLayer, self).__init__()

        # 多头自注意力机制
        self.attention_A = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attention_B = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # LayerNorm 和 Dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        :param x: [x1, x2, x3, x4] xi形状 (batch_size, channel, h, w)
        :return: 经过 Transformer Layer 处理后的输出
        """
        output_list = []
        
        for layer in x:
            layer = layer.view(layer.size(0), layer.size(1), -1)
            layer_A, layer_B = layer[:layer.size(0)//2].permute(0, 2, 1), layer[layer.size(0)//2:].permute(0, 2, 1) # [B, HW, C]
            attn_output_A, _ = self.attention_A(layer_A, layer_A, layer_A)  # Q, K, V 相同 # [B, HW, C]
            attn_output_B, _ = self.attention_B(layer_B, layer_B, layer_B)  # Q, K, V 相同 # [B, HW, C]

            attn_output = torch.maximum(attn_output_A, attn_output_B)
            
            
            x_A = F.adaptive_avg_pool1d((attn_output + self.dropout1(attn_output)).permute(0, 2, 1), 1).squeeze(-1)  # [B, C]

            feat = x_A # [B, C]
            output_list.append(feat)
        out = torch.cat(output_list, dim=-1)
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, embed_dim=128, num_heads=8, ff_dim=128, dropout=0.1):
        """
        多层 Transformer Encoder
        :param num_layers:  Transformer 层的数量
        :param embed_dim:   输入特征维度 (d_model)
        :param num_heads:   多头注意力的头数
        :param ff_dim:      前馈网络的隐藏层维度
        :param dropout:     Dropout 概率
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状 (batch_size, seq_len, embed_dim)
        :return: 经过多层 Transformer 处理后的输出
        """
        for layer in self.layers:
            x = layer(x)
        return x







