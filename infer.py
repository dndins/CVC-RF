import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import DataSet
from torchvision import transforms
from tqdm import tqdm
import torchvision
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from utils import *
from collections import Counter

from train_End2End import *
from itertools import zip_longest
import json
import time
from sklearn.metrics import roc_auc_score

transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def calculate_metrics(y_true, y_pred):
    """
    计算 Sensitivity, Specificity, F1, PPV, NPV
    
    参数:
    y_true: 真实标签列表
    y_pred: 预测标签列表
    
    返回:
    包含所有指标的字典
    """
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算各项指标
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # PPV (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # 使用sklearn的classification_report获取更多指标
    report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics = {
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'PPV (Precision)': ppv,
        'NPV': npv,
        'F1 Score': f1,
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }
    
    return metrics





def infer(net, device, json_path, root_path, module, weight_path):
    
    device = torch.device(device)
    net.to(device)
    
    valid_d = DataSet(json_path, transform=transform, target='test')
    valid_load = DataLoader(valid_d, batch_size=1, shuffle=True, num_workers=1)
    
    iterations = len(valid_d)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    random_input = torch.randn(1, 3, 224, 224).to(device)
    va = torch.ones((4, 128)).to(device)
    vb = torch.ones((4, 128)).to(device)
    spacing = torch.tensor(1.0).to(device)
    # for _ in range(50):
    #      _, _, _ = net(random_input, spacing, va, vb)
    
    torch.cuda.synchronize()  # 等待GPU计算完成（很重要）

    pred_labels = []
    feature_list = []
    pred_list = []
    label_list = []
    name_list = []
    incorreect_dict = {}
    with torch.no_grad():
        net.eval()
        
        pred_dict = {}

        for i, (img_A, img_B, labels, ids, name, spacing, _) in enumerate(valid_load):
            img_A, img_B, labels, spacing = img_A.to(device), img_B.to(device), labels.to(device), spacing.to(device)
            img = torch.cat([img_A, img_B], dim=0)

            va = torch.ones((2, 128)).to(device)
            vb = torch.ones((2, 128)).to(device)

            xa, xb, preds = net(img, spacing, va ,vb)
            mid = torch.cat([xa, xb], dim=-1)

            prob = torch.softmax(preds, dim=-1)  # 概率 (batch_size, n_classes)
            # score, pred = torch.max(prob, dim=1)     # 预测类别
            # pred = prob[:, 1] > 0.268945
            pred = prob[:, 1] > 0.5
            # 保留错误样本信息
            if pred != labels:
                incorreect_dict[name[0]] = [
                    int(pred + 1),
                    [round(num, 3) for num in prob[0].tolist()]
                ]

            feature_list.append(mid)

            # 存真实标签
            label_list += labels.cpu().tolist()
            pred_labels += pred.cpu().tolist()
            # 存概率
            pred_list.append(prob[:, 1].cpu().detach().numpy())  # 先存成数组形式
            
            pred_dict[name[0].split('/')[-1]] = {"gt":int(labels[0].cpu().numpy()),
                                  "pred":int(pred.cpu().numpy()),
                                  "score":[float(prob[0,0].cpu().numpy()), float(prob[0,1].cpu().numpy())]}
        
        with open(opt.save_path + r'/test_record_outside.json', 'w', encoding='utf-8') as f:
            json.dump(pred_dict, f, indent=4)

        pred_array = np.vstack(pred_list)  # (n_samples, n_classes)
        auc = roc_auc_score(label_list, pred_array)
        print("AUC:", auc)
        
        metrics = calculate_metrics(label_list, pred_labels)

        print("评估指标:")
        for metric, value in metrics.items():
            if metric in ['TP', 'TN', 'FP', 'FN']:
                print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value:.4f}")
        print(confusion_matrix(label_list, pred_labels))
        # 记录验证信息
        tqdm.write(classification_report(label_list, pred_labels, labels=None, target_names=None, sample_weight=None, digits=3, output_dict=False))
        report = classification_report(label_list, pred_labels, labels=None, target_names=None, sample_weight=None, digits=3, output_dict=False)
        
        cm = confusion_matrix(label_list, pred_labels)
        mcc = average_mcc(label_list, pred_labels, num_classes=3)
        class_acc = cm.diagonal() / cm.sum(axis=1)
    
    
        record_txt(incorreect_dict, root_path, json_path, module, epochs=None, batch_size=None, lr=None, weight_path=None, report=report, class_acc=class_acc, cm=cm, mean_acc =mcc, infer=True)
        
    



if __name__ == '__main__':
    set_seed(42)  # 42是示例种子，可以选择任何整数
    opt = para()
    if opt.module == 'vit-b':
       net = torchvision.models.vit_b_16(weights=True)
       net.heads = nn.Sequential(nn.Linear(net.heads.head.in_features, 3))
                                
    elif opt.module == 'ResNet18_2':
        net = NetWork(num_class=opt.num_class)
    net.load_state_dict(torch.load(opt.test_weight_path))
    

    infer(net, opt.device, opt.json_path, opt.save_path, opt.module, opt.weight_path)



