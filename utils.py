import random
from torch import nn
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import numpy as np
from torchvision.models import resnet18

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------Memory Bank-----------------------------------#
class MemoryBank:
    def __init__(self, size, dim, alpha=0.99, class_num=4, device='cuda'):
        self.size = size
        self.dim = dim
        self.alpha = alpha
        self.device = device
        self.class_num = class_num
        self.net = resnet18(pretrained=True)
        input_num = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(input_num, 128))
        self.net.to(device)
        self.bank = torch.zeros(size, dim, device=device)
        self.labels = torch.zeros(size).to(device)
    
    # 初始化memory_bank 并加载初始向量
    def get_init_memory_bank(self, data_loader, device, view):
        # 输入数据加载器  设备 视角类别
        
        
        with torch.no_grad():
            self.net.eval()
            # 将
            for i, (img_A, img_B, label, inst_id, _, spacing,_) in enumerate(data_loader):
                # 图像A 图像B 图像标签 图像在序列的位置
                img_A, img_B, label = img_A.to(device), img_B.to(device), label.to(device)
                
                # memory bank 中向量对应的标签
                self.labels[inst_id] = label.float()
                
                if view == 'A':
                    out = self.net(img_A)
                    out = F.normalize(out[0], dim=-1)
                    self.bank[inst_id] = out
                else:
                    out = self.net(img_B)
                    out = F.normalize(out[0], dim=-1)
                    self.bank[inst_id] = out


    # 更新bank
    def update(self, indices, features):
        features = torch.nn.functional.normalize(features, dim=1)  # Normalize features
        # Fetch old vectors from the memory bank
        old_features = self.bank[indices]
        # EMA update
        updated_features = self.alpha * old_features + (1 - self.alpha) * features
        updated_features = torch.nn.functional.normalize(updated_features, dim=1)
        # Store updated features back into the memory bank
        self.bank[indices] = updated_features


    # 按照索引寻找对应向量
    def get_features(self, indices):
        return self.bank[indices]
    
    
    # 获得每类的中心向量
    def get_class_mean_feature(self):
        vectors_list = []
        for i in range(self.class_num):
            mean_feature = self.bank[self.labels == i, :].mean(dim=0)
            mean_feature = torch.nn.functional.normalize(mean_feature, dim=-1)
            vectors_list.append(mean_feature)
        return vectors_list

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[2:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        
        
    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        xx = F.softmax(preds, dim=1)
        xx = torch.clamp(xx, min=1e-4, max=1.0)
        preds_logsoft = torch.log(xx)  # softmax后取对数 [-00, 0]
        

        preds_softmax = torch.exp(preds_logsoft) # [0, 1]
        
        
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        
        return loss
    
class MLP_Head(nn.Module):
    def __init__(self, net, in_dim, out_dim):
        super(MLP_Head, self).__init__()
        self.net = net
        self.fc_0 = nn.Linear(1000, in_dim)
        self.fc_1 = nn.Linear(in_dim * 2, 256)
        self.fc_2 = nn.Linear(256, out_dim)
    
    def forward(self, x):
        x = self.net(x)
        x = self.fc_0(x)
        x1 = x[:x.size(0) // 2]
        x2 = x[x.size(0) // 2:]
        x = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x3))
        return x, x1, x2

class MLP_Recovey_Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_Recovey_Head, self).__init__()
        self.fc_1 = nn.Linear(in_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 128)
        self.fc_4 = nn.Linear(128, out_dim)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))
        x = F.leaky_relu(self.fc_4(x))
        return x
    
class MLP_infer_Head(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(MLP_infer_Head, self).__init__()
        self.fc_1 = nn.Linear(in_dim, 128)
        self.fc_2 = nn.Linear(128, out_dim)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        return x

def record_txt(incorrect, root_path, json_path, module, epochs, batch_size, lr, weight_path, report, class_acc, cm, mean_acc, infer=False):
    if infer == False:
        with open(root_path + '/best{}.txt'.format(json_path.split('/')[-1][:-5]), 'w', encoding="utf-8") as f:
            f.write('net: ' + module + '\nepochs: ' + str(epochs) + '\nbatch_size: ' + str(batch_size) + '\nlr: ' +  str(lr)
                    + '\nweight_path: ' + weight_path + '\njson_path: ' + json_path + '\nroot_path: ' + root_path + '\n')
            f.write(report + '\n')
            for x in range(len(class_acc)):
                f.write('acc{}: {}\n'.format(x, class_acc[x]))

            f.write('best_mean_acc:{}\n'.format(class_acc.mean()))
            f.write('mean_mcc:{}\n'.format(mean_acc))
            f.write('confusion_matrix:\n {}\n'.format(cm))
    if infer == True:
        with open(root_path + '/best{}.txt'.format(json_path.split('/')[-1][:-5]), 'a', encoding="utf-8") as f:
            f.write('\n\n\nInfer: \n')
            f.write(report + '\n')
            for x in range(len(class_acc)):
                f.write('acc{}: {}\n'.format(x, class_acc[x]))
                
            f.write('best_mean_acc:{}\n'.format(class_acc.mean()))
            f.write('mean_mcc:{}\n'.format(mean_acc))
            f.write('confusion_matrix:\n {}\n'.format(cm))
           
            for name, label in incorrect.items():
                f.write('{}: {}\n'.format(name, label))
    

def record_plot(epochs, loss_train, loss_valid, total_acc_valid, valid_acc_list, root_path):
    # 绘制损失函数
    titles = ['Loss_train', 'Loss_valid']
    y_list = [loss_train, loss_valid]

    plt.figure(figsize=(10, 5))  # 创建画布并设置大小

    # 绘制两条曲线
    plt.plot(range(epochs), y_list[0], label=titles[0], color='blue', linewidth=2)  # 训练损失曲线
    plt.plot(range(epochs), y_list[1], label=titles[1], color='orange', linewidth=2)  # 验证损失曲线

    # 添加标题、标签和图例
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')  # 在右上角显示图例

    # 保存图片
    plt.savefig(root_path + '/plot_loss.png')
    
    # 绘制精度函数
    titles = ['total_acc_valid', 'acc1', 'acc2', 'acc3', 'acc4']
    y_list = [total_acc_valid, valid_acc_list[0], valid_acc_list[1], valid_acc_list[2], valid_acc_list[3]]
    
    plt.figure(2, figsize=(15, 10))
    for i in range(len(y_list)):
        plt.subplot(231+i)
        plt.plot(range(epochs), y_list[i])
        plt.title(titles[i])
    plt.savefig(root_path + f'/polt_acc.png')
    
class class_Head(nn.Module):
    def __init__(self, net, in_dim, out_dim):
        super(class_Head, self).__init__()
        self.net = net
        self.fc_1 = nn.Linear(in_dim, 128)
        self.fc_2 = nn.Linear(128, out_dim)
    
    def forward(self, x):
        xs = self.net(x)
        
        x1 = xs[:xs.size(0) // 2]
        x2 = xs[xs.size(0) // 2:]
        # concate
        x = torch.cat((x1, x2), dim=1)

        x = torch.relu(self.fc_1(x))
        x3 = torch.relu(self.fc_2(x))
        return xs, x3
    
class MOE_Head(nn.Module):
    def __init__(self, net, in_dim, out_dim):
        super(MOE_Head, self).__init__()
        
        self.backbone = net
        
        self.expert1 = nn.Sequential(nn.Linear(in_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, out_dim))
        
        self.expert2 = nn.Sequential(nn.Linear(in_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, out_dim))
        
        
        self.router = nn.Sequential(nn.Linear(in_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, 2))
        
    def forward(self, x):
        x_bb = self.backbone(x)
        xa = x_bb[:x_bb.size(0)//2]
        xb =  x_bb[x_bb.size(0)//2:]
        x = torch.cat((xa,xb), dim=1)
        
        out1 = self.expert1(x)
        out2 = self.expert2(x)
        weight = F.softmax(self.router(x), dim=1)
        
        out = torch.stack((out1, out2)).permute(1, 0, 2)
        out = (out * weight.unsqueeze(-1)).sum(dim=1)
        return x_bb, out, weight

def average_mcc(y_true, y_pred, num_classes):
    """
    计算多分类任务的平均 Matthews 相关系数 (MCC)
    
    参数:
    - y_true: list 或 numpy 数组，真实标签
    - y_pred: list 或 numpy 数组，预测标签
    - num_classes: int，类别的数量
    
    返回:
    - average_mcc: float，多分类任务的平均 Matthews 相关系数
    """
    mcc_list = []
    
    # 遍历每个类别，计算二分类 MCC
    for i in range(num_classes):
        # 将当前类别视为正类，其他类别视为负类
        y_true_binary = (np.array(y_true) == i).astype(int)
        y_pred_binary = (np.array(y_pred) == i).astype(int)
        
        # 计算当前类别的 MCC
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        mcc_list.append(mcc)
    
    # 计算平均 MCC
    average_mcc = np.mean(mcc_list)
    return average_mcc


def for_and_backward_block(Memory_bank, query, labels, ids, criterion_nce=None, train=True):
    
    query = F.normalize(query, dim=-1)
    
    # 计算每个类别的聚类中心
    center_vectors = Memory_bank.get_class_mean_feature() # [4, 128]
    matrix_center_vectors = torch.stack(center_vectors).unsqueeze(0).repeat(query.size(0), 1, 1).to(query.device) # [32, 4, 128]
    
    positive_key = matrix_center_vectors[torch.arange(query.size(0)), labels]
    positive_key = F.normalize(positive_key, dim=-1)
    
    negative_keys = []
    
    for class_num in labels:
        neg_idx = Memory_bank.labels != class_num
        negative_key = Memory_bank.get_features(neg_idx)
        negative_keys.append(negative_key)
    
    # 将预测向量 正例中心向量 标签 以及对应的聚类中心向量输入 计算nce loss
    loss = criterion_nce(query, positive_key, negative_keys)  
    if train == True:
    # 用预测向量更新Memory_bank
       Memory_bank.update(ids, query.detach())
    return loss


def polt_tsne(feature_list, label_list, save_path):

    label_dict = {0:'RADS1_pred',
                  1:'RADS2_pred',
                  2:'RADS3_pred',
                  3:'RADS4_pred',
                  4:'RADS1_Label',
                  5:'RADS2_Label',
                  6:'RADS3_Label',
                  7:'RADS4_Label',}
    
    features = torch.stack(feature_list).cpu().numpy()
    labels = np.array(label_list)
    save_array = np.concatenate((features, np.expand_dims(labels, axis=1)), axis=1)
    print(save_array.shape)
    np.save('/nvme/ccy/zzy_projects/Carotid_Project/GCN_Layer/train.npy', save_array)
    
    print(features.shape)
    print(labels.shape)

    # 计算 t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features)

    color_bar = {0: plt.cm.tab20(0), 1: plt.cm.tab20(2), 2: plt.cm.tab20(14), 3:plt.cm.tab20(8), 4:plt.cm.tab20(1), 5:plt.cm.tab20(3), 6:plt.cm.tab20(5), 7:plt.cm.tab20(7)} # 10


    colors = [color_bar[x] for x in labels]

    # 可视化 t-SNE 结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], s=12, c=colors)  

    plt.xticks(range(50, 50, 25))
    plt.yticks(range(50, 50, 25))


    # handles, _ = scatter.legend_elements()
    # legend_labels = [label_dict[i] for i in range(6)]
    # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_bar[i], markersize=10) for i in range(6)]
    # plt.legend(handles, legend_labels, title="Classes")

    # plt.savefig('/mnt/data1/zzy/ProjecT/DYF_US_Synthesis/gen_tsne.png')
    
    # Create custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_bar[i], markersize=10) for i in range(len(label_dict))]
    legend_labels = [label_dict[i] for i in range(len(label_dict))]
    plt.legend(handles, legend_labels, title="Classes")
    
    
    # Save the figure
    plt.savefig(save_path + f'/tsne.png', dpi=300)
    
class MOE_Hard_Gate(nn.Module):
    def __init__(self, backbone, input_dim, num_classes):
        super(MOE_Hard_Gate, self).__init__()
        self.backbone = backbone
        
        # 定义 Gate 函数
        self.gate = GateFunction(input_dim)
        
        # 定义两个专家网络
        self.expert_a = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 专家 A 输出类别 0 和 1 的 logits
        )
        self.expert_b = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 专家 B 输出类别 2 和 3 的 logits
        )
    
    def forward(self, x):
        
        x = self.backbone(x)
        x = torch.cat((x[:x.size(0)//2], x[x.size(0)//2:]), dim=1)
        
        # 获取每个输入的专家分配索引
        gate_indices = self.gate(x)  # [batch_size]
        
        # 初始化专家输出
        batch_size = x.size(0)
        device = x.device
        expert_output = torch.zeros(batch_size, 4, device=device)  # [batch_size, num_classes]
        
        # 专家 A 的索引和处理
        mask_a = (gate_indices == 0)  # [batch_size], 属于专家 A 的样本
        if mask_a.any():
            output_a = self.expert_a(x[mask_a])  # [num_a, 2]
            expert_output[mask_a, :2] = output_a  # 填充类别 0 和 1 的结果
        
        # 专家 B 的索引和处理
        mask_b = (gate_indices == 1)  # [batch_size], 属于专家 B 的样本
        if mask_b.any():
            output_b = self.expert_b(x[mask_b])  # [num_b, 2]
            expert_output[mask_b, 2:] = output_b  # 填充类别 2 和 3 的结果
        
        return expert_output
    

class GateFunction(nn.Module):
    def __init__(self, input_dim):
        super(GateFunction, self).__init__()
        # Gate 网络，用于决定分配到哪个专家
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出两个专家的 logits
        )

    def forward(self, x):
        # 硬分配：返回每个输入对应的专家索引（0 或 1）
        gate_logits = self.gate_net(x)  # [batch_size, 2]
        gate_indices = torch.argmax(gate_logits, dim=1)  # [batch_size]
        return gate_indices


class Overfit_Loss(nn.Module):
    def __init__(self, delta=0., lambda_=1.0):
        """
        自定义损失函数
        Args:
            delta (float): 限制置信度的最大值，默认值为 0.8。
            lambda_ (float): 惩罚项权重。
        """
        super(Overfit_Loss, self).__init__()
        self.delta = delta
        self.lambda_ = lambda_
        self.ce_loss = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): 模型的 softmax 输出，形状为 (batch_size, num_classes)。
            targets (Tensor): 真实标签，形状为 (batch_size,)。
        Returns:
            Tensor: 总损失值。
        """
        # 计算交叉熵损失
        ce_loss = self.ce_loss(outputs, targets)
        
        # 获取目标类别的 softmax 概率
        probs = torch.softmax(outputs, dim=1)
        target_probs = probs[range(len(targets)), targets]  # 获取每个样本的目标概率

        # 计算正则化项
        reg_loss = torch.clamp(target_probs - self.delta, min=0) ** 2
        reg_loss = reg_loss.mean()  # 平均化

        # 总损失
        total_loss = ce_loss + self.lambda_ * reg_loss
        return total_loss

class Diff_Feat_Head(nn.Module):
    def __init__(self, net, in_dim, out_dim):
        super(Diff_Feat_Head, self).__init__()

        self.backbone = net
      
        self.expert1 = nn.Sequential(nn.Linear(in_dim//2, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, out_dim))

        self.expert2 = nn.Sequential(nn.Linear(in_dim//2, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, out_dim))
        
        self.expert3 = nn.Sequential(nn.Linear(in_dim//2, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, out_dim))


        self.router = nn.Sequential(nn.Linear(in_dim*2, 128),
                                nn.ReLU(),
                                nn.Linear(128, 2))
        
    def forward(self, x):
        x_bb = self.backbone(x)
        
        xa = x_bb[:x_bb.size(0)//2]
        xb =  x_bb[x_bb.size(0)//2:]
        
        feat_1 = xa[:, :xa.size(1)//2]
        feat_2 = xb[:, :xb.size(1)//2]
        # feat_3 = torch.cat((xa[:, xa.size(1)//2:], xb[:, xb.size(1)//2:]), dim=1)
        feat_3 = xa[:, xa.size(1)//2:] + xb[:, xb.size(1)//2:]
        
        x_weight = torch.cat((xa,xb), dim=1)
        
        out1 = self.expert1(feat_1)
        out2 = self.expert2(feat_2)
        out3 = self.expert3(feat_3)
        
        weight = F.softmax(self.router(x_weight), dim=1)
        print(weight[0].cpu().tolist())
        out = torch.stack((out1, out2)).permute(1, 0, 2)
        out = (out * weight.unsqueeze(-1)).sum(dim=1)
        out = out + out3
        return xa, xb, out



def custom_infonce_loss(A, B, epoch, temperature=0.07):
    """
    自定义 InfoNCE 损失：
    - A1 和 B1 是正样本对。
    - 其他组合 (A1, B2), (A2, B1), (A2, B2) 是负样本对。
    
    Args:
        A (Tensor): 第一个特征向量 (batch_size, 128)。
        B (Tensor): 第二个特征向量 (batch_size, 128)。
        temperature (float): 温度参数 τ。
    Returns:
        Tensor: 损失值。
    """
    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    batch_size = A.size(0)
    
    # 切分 A 和 B
    A1, A2 = A[:, :64], A[:, 64:]
    B1, B2 = B[:, :64], B[:, 64:]

    # 计算正样本对相似性
    pos_sim = F.cosine_similarity(A1, B1, dim=-1)  # (batch_size,)
    
    # 计算负样本对相似性
    neg_sim_A1_B2 = torch.matmul(A1, B2.T)  # (batch_size, batch_size)
    neg_sim_A2_B1 = torch.matmul(A2, B1.T)  # (batch_size, batch_size)
    neg_sim_A2_B2 = torch.matmul(A2, B2.T)  # (batch_size, batch_size)

    # 对每个样本构造 logits
    logits = torch.cat([
        pos_sim.unsqueeze(1),  # 正样本相似性 (batch_size, 1)
        neg_sim_A1_B2,
        neg_sim_A2_B1,
        neg_sim_A2_B2
    ], dim=1) / temperature  # 缩放 logits

    # 构造标签：正样本是第一个位置
    labels = torch.zeros(batch_size, dtype=torch.long).to(A.device)

    # 计算交叉熵损失
    loss = F.cross_entropy(logits, labels)
    return loss

def contrastive_loss(A, B, temperature=0.1):
    """
    计算自定义对比损失：
    A[i, :64] 的正对是 B[i, :64]，负对是 B[j, :64] (j != i)。

    参数:
        A: Tensor [n, 128]，输入向量 A
        B: Tensor [n, 128]，输入向量 B
        temperature: 温度参数

    返回:
        loss: 对比损失
    """
    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    
    bs = A.size(0)
    n = A.size(1)

    # 取前 64 维特征
    A_1 = A[:, :n//2]  # [n, 64]
    B_1 = B[:, :n//2]  # [n, 64]
    A_2 = A[:, n//2:]  # [n, 64]
    B_2 = B[:, n//2:]  # [n, 64]

    # 计算余弦相似度矩阵
    logits_up = torch.matmul(A_1, B_1.T)  # [32, 32]
    logits_up /= temperature  # 缩放
    logits_down = torch.matmul(A_1, B_2.T) # [32, 32]
    logits_down /= temperature  # 缩放
    

    # 为负对创建掩码
    mask = torch.eye(bs, dtype=torch.bool).to(A.device)  # [n, n] 对角线为 True
    logits_pos = logits_up[mask].unsqueeze(1)
    logits_neg1 = logits_up[~mask].view(bs, bs-1)
    logits_neg2 = logits_down
    
    logits = torch.cat([logits_pos, logits_neg1, logits_neg2], dim=-1)
    labels = torch.zeros(bs, dtype=torch.long).to(A_1.device)
    # 计算 Softmax 后的 InfoNCE 损失
    loss = F.cross_entropy(logits, labels)

    return loss
