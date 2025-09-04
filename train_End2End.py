import os
import torch
import torchvision
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms.functional
from dataset import DataSet, balance_sampler
from torchvision import transforms
from tqdm import tqdm

import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from net_worker import NetWork
from losses import *
from utils import MemoryBank, for_and_backward_block, average_mcc, record_txt, record_plot, set_seed


transform = {'train': transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.5, hue=0.5),
                                transforms.ToTensor(),        
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
            'valid': transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
            }



def main(net, epochs, batch_size, lr, device, weight_path, json_path, root_path, module, opt):
    
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    
    device = torch.device(device)
    net.to(device)
    
    # 损失函数
    criterion_ce = nn.CrossEntropyLoss()
    criterion_nce = Sup_InfoNCE()
    

    optimizer = torch.optim.AdamW(net.parameters(),  lr)

    lr_step = CosineAnnealingLR(optimizer, T_max=epochs//4, eta_min=lr*0.01)
    

    train_d = DataSet(json_path, transform=transform['train'], target='train')
    labels = [train_d[i][2].item() for i in range(len(train_d))]
    sampler = balance_sampler(labels, batch_size=6, num_batches=500)
    train_load = DataLoader(train_d, batch_sampler=sampler, num_workers=8)


    valid_d = DataSet(json_path, transform=transform['valid'], target='valid')
    valid_load = DataLoader(valid_d, batch_size=batch_size, shuffle=True, num_workers=1)

    # 初始化bank
    bank_A = MemoryBank(size=len(train_d), dim=128, class_num=opt.num_class, device=device)
    bank_B = MemoryBank(size=len(train_d), dim=128, class_num=opt.num_class, device=device)
    
    bank_A.get_init_memory_bank(data_loader=train_load, device=device, view='A')
    bank_B.get_init_memory_bank(data_loader=train_load, device=device, view='B')

    
    # 记录训练集的损失 验证集的损失
    loss_train, loss_valid = [], []
    
    # 记录验证集的精度
    total_acc_valid = []
    # 记录训练集每个类的精度 验证集每个类的精度
    valid_acc_list = [[], [], [], []]
    
    # 记录最好平均精度和整体精度
    best_mean_acc = 0
    best_acc = 0
    
    for epoch in tqdm(range(epochs), ncols=100, colour='WHITE'):
        train_loss = 0
        valid_loss = 0
        pred_list = []
        label_list = []
        
        for img_A, img_B, labels, ids, _, spacing,_ in train_load:
            # print(img_A.shape)
            # 图像A 图像B 图像标签 图像在Memory Bank的位置
            optimizer.zero_grad()
            img_A, img_B, labels, spacing = img_A.to(device), img_B.to(device), labels.to(device), spacing.to(device)
            img = torch.cat([img_A, img_B], dim=0)
            
            va = torch.stack(bank_A.get_class_mean_feature())
            vb = torch.stack(bank_B.get_class_mean_feature())

            # 获得输出结果
            xa, xb, out = net(img, spacing, va, vb)

            # flops, params = profile(net, inputs=(img, spacing, va, vb)) # thop 计算结果
            # tqdm.write(f"FLOPs: {flops/1e9:.2f} GFLOPs") # 转换为十亿级（GFLOPs

            # 分别对两个分支做损失 更新bank
            loss_nce_A= for_and_backward_block(Memory_bank=bank_A, query=xa, labels=labels, ids=ids, criterion_nce=criterion_nce)
            loss_nce_B= for_and_backward_block(Memory_bank=bank_B, query=xb, labels=labels, ids=ids, criterion_nce=criterion_nce)
            
            loss_nce = (loss_nce_A + loss_nce_B)/2
            
            loss_focal = criterion_ce(out, labels)
            # if epoch <=20:
            #     loss =  loss_nce
            # else:
            loss = loss_focal
            # tqdm.write(f'loss_ce:{loss_focal}')
            # tqdm.write(f'loss_nce:{loss_nce*0.2}')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        lr_step.step()
      
        mean_train_loss = train_loss / len(train_load)
        loss_train.append(mean_train_loss)
        
        with torch.no_grad():
            net.eval()
            for i, (img_A, img_B, labels, ids, _, spacing, _) in enumerate(valid_load):
                # 图像A 图像B 图像标签 图像在序列的位置
                img_A, img_B, labels, spacing = img_A.to(device), img_B.to(device), labels.to(device), spacing.to(device)
                # 输入图像 获得预测向量 
                img = torch.cat([img_A, img_B], dim=0)
                
                va = torch.stack(bank_A.get_class_mean_feature())
                vb = torch.stack(bank_B.get_class_mean_feature())
                
                xa, xb, pred = net(img, spacing, va, vb)

                
                # 只计算 不更新
                loss_nce_A = for_and_backward_block(Memory_bank=bank_A, query=xa, labels=labels, ids=ids, criterion_nce=criterion_nce, train=False)
                loss_nce_B= for_and_backward_block(Memory_bank=bank_B, query=xb, labels=labels, ids=ids, criterion_nce=criterion_nce, train=False)
                
                loss_nce = (loss_nce_A + loss_nce_B)/2
                
                loss_focal = criterion_ce(pred, labels)
                
                # if epoch <= 20:
                #     loss =  loss_nce
                # else:
                loss = loss_focal


                valid_loss += loss.item()

                # 保留验证结果
                pred = torch.softmax(pred, dim=-1)
                pred = torch.argmax(pred, dim=1)
                pred_list += pred.cpu().tolist()
                label_list += labels.cpu().tolist()
            
            mean_valid_loss = valid_loss / len(valid_load)
            loss_valid.append(mean_valid_loss)
            

            # 记录验证信息
            tqdm.write(classification_report(label_list, pred_list, labels=None, target_names=None, sample_weight=None, digits=3, output_dict=False))
            report = classification_report(label_list, pred_list, labels=None, target_names=None, sample_weight=None, digits=3, output_dict=False)
            all_acc = accuracy_score(label_list, pred_list)
            mean_MCC = average_mcc(label_list, pred_list, opt.num_class)
            total_acc_valid.append(all_acc)
            cm = confusion_matrix(label_list, pred_list)
            
            class_acc = cm.diagonal() / cm.sum(axis=1)
            for i, acc in enumerate(class_acc):
                tqdm.write(f"Class {i} Accuracy: {acc:.3f}")
                valid_acc_list[i].append(acc)
        
        
            # 记录整体最高精度
            if all_acc > best_acc:
                best_acc = all_acc
                
            tqdm.write(f"best_acc: {best_acc:.3f}")
            tqdm.write(f"mean_mcc: {mean_MCC:.3f}")
           
            tqdm.write(f"epoch: {epoch}       train loss: {mean_train_loss:.3f}")
            tqdm.write(f"epoch: {epoch}       valid loss: {mean_valid_loss:.3f}")
            
            # 记录平均最高精度 当平均精度最大时 保留记录
            if np.mean(class_acc) > best_mean_acc:
                best_mean_acc = np.mean(class_acc)
                torch.save(net.state_dict(), weight_path + '/best.pth')
                record_txt(None, root_path, json_path, module, epochs, batch_size, lr, weight_path, report, class_acc, cm, mean_MCC)
            
            if epoch % 10 == 0:
                torch.save(net.state_dict(), weight_path + '/epoch{}.pth'.format(epoch))
                


    # 画图
    record_plot(epochs, loss_train, loss_valid, total_acc_valid, valid_acc_list, root_path)
    



def para():
    arg = argparse.ArgumentParser() 
    arg.add_argument('--module', type=str, default='ResNet18_2')
    arg.add_argument('--epochs', type=int, default=100)
    arg.add_argument('--batch_size', type=int, default=4)
    arg.add_argument('--lr', type=float, default=1e-4)
    arg.add_argument('--num_class', type=int, default=2)
    arg.add_argument('--device', type=str, default='cuda:4')
    arg.add_argument('--root_path', type=str, default='/mnt/data1/zzy/ProjecT/B_ProJ_bank_mvit/')
    arg.add_argument('--json_file', type=str, default='Norm_Stable_Vulnerable_Crop_add7.json')
    # arg.add_argument('--json_file', type=str, default='Outside_test.json')
    
    # 测试权重
    arg.add_argument('--test_weight_path', type=str, default= '/mnt/data1/zzy/ProjecT/B_ProJ_bank_mvit/ResNet18_2_Norm_Stable_Vulnerable_class2_clstrain/weight/best.pth')
    opt = arg.parse_args()
    
    # 保存信息文件夹
    opt.save_path = opt.root_path + opt.module + '_Norm_Stable_Vulnerable_class2_clstrain'
    os.makedirs(opt.save_path, exist_ok=True)
    opt.weight_path = opt.save_path + '/weight'
    os.makedirs(opt.weight_path, exist_ok=True)
    opt.json_path = opt.root_path + opt.json_file 
    return opt

if __name__ == '__main__':
    set_seed(42) 
    opt = para()
         
    if opt.module == 'ResNet18_2':
       net = NetWork(num_class=opt.num_class)
       net.load_state_dict(torch.load(r'/mnt/data1/zzy/ProjecT/B_ProJ_bank_mvit/ResNet18_2_Norm_Stable_Vulnerable_class2_pretrain/weight/best.pth'))
    #    for name, param in net.named_parameters():
    #        if 'simple_class' not in name:
    #            param.requires_grad = False

    # total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params}")
    

    main(net, opt.epochs, opt.batch_size, opt.lr, 
         opt.device, opt.weight_path, opt.json_path, opt.save_path, opt.module, opt)