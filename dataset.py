import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

class DataSet(Dataset):
    def __init__(self, json_path, transform, target='path'):
        super(Dataset, self).__init__()
        self.root_path = json_path
        self.transform = transform
        self.target= target
        self.targets = []
       
        self.spacing = json.load(open(r'/mnt/data1/zzy/ProjecT/B_ProJ_bank_mvit/data_spacing.json', 'r', encoding='utf-8'))
        if target != "test_outside":
            
            # # 数据平均
            self.path_list_1 = [x for x in json.load(open(json_path, 'r', encoding='utf-8'))[target] if "CLASS1" in x]
            self.path_list_2 = [x for x in json.load(open(json_path, 'r', encoding='utf-8'))[target] if "CLASS2" in x]
            self.path_list_3 = [x for x in json.load(open(json_path, 'r', encoding='utf-8'))[target] if "CLASS3" in x]
            if target == "train":
                self.path_list = self.path_list_2 + self.path_list_3
                print(f'数据初始化完成：Class1数据量：{len(self.path_list_1)} Class12数据量：{len(self.path_list_2)} Class3数据量：{len(self.path_list_3)}')
            else:
                self.path_list = self.path_list_2 + self.path_list_3
            
            # for x in self.path_list:
            #     if 'CLASS1' in x:
            #         self.targets.append(0)
            #     elif 'CLASS2' in x:
            #         self.targets.append(1)
            #     elif 'CLASS3' in x:
            #         self.targets.append(2)

            for x in self.path_list:
                if 'CLASS2' in x:
                    self.targets.append(0)
                elif 'CLASS3' in x:
                    self.targets.append(1)
        else:
            self.dicts = json.load(open(json_path, 'r', encoding='utf-8'))
            self.path_list = [x for x in self.dicts]
        


            for x in self.path_list:
                if 'RADS2' in x:
                    self.targets.append(0)
                elif 'RADS3' in x or 'RADS4' in x :
                    self.targets.append(1)

        # 每一个样本的序列
        self.instances_id = [i for i in range(len(self.path_list))]   
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        # 图像路径
        if self.target == "test_outside":
            img_A_path = os.path.join(r'/mnt/data1/zzy/Data/Outside_test_crop/Plane_A',self.path_list[idx])
            img_B_path = os.path.join(r'/mnt/data1/zzy/Data/Outside_test_crop/Plane_B',self.dicts[self.path_list[idx]])
        else:
            img_A_path = self.path_list[idx]
            img_B_path = img_A_path.replace('_A', '_B')


        # 该图像在序列中的位置
        instance_id = self.instances_id[idx]
        # 该图像的标签
        label = self.targets[idx]
        
        key = img_A_path.split('/')[-1]
        if 'RADS' not in key:
            key = f'RADS{int(label)+1}_' + img_A_path.split('/')[-1]
        
        spacing_keys = self.spacing.keys()
        
        if key in spacing_keys:
            spacing = self.spacing[key]
        else:
            spacing = 0
        spacing = torch.tensor(spacing).float()

        if not os.path.exists(img_A_path):
            print(img_A_path)
        if not os.path.exists(img_B_path):
            print(img_B_path)
        
        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        
        H_A, W_A = img_A.size
        H_B, W_B = img_B.size
        
        img_A = img_A.convert("RGB")
        img_A = self.transform(img_A).float()
        
        img_B = img_B.convert("RGB")
        img_B = self.transform(img_B).float()
        
        label = torch.tensor(label)
        return img_A, img_B, label, instance_id, img_A_path, spacing, [H_A, W_A, H_B, W_B]

    def pad_to_square(self, img, fill_color=(0, 0, 0)):

            # 获取图像的宽度和高度
            width, height = img.size
            
            # 计算正方形的边长
            new_size = max(width, height)
            
            # 创建一个新的正方形图像，并将原图粘贴到中心
            padded_img = Image.new("RGB", (new_size, new_size), fill_color)
            padded_img.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
            
            return padded_img
    
from collections import defaultdict
import random

class balance_sampler(Sampler):
    def __init__(self, labels, batch_size, num_batches=None):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = len(set(labels)) 
        assert batch_size % self.num_classes == 0, "Batch size must equal number of classes for 1 sample per class."

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_batches = num_batches if num_batches is not None else 1000  # 设置一个默认的 batch 数目

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for cls in self.classes:
                batch.append(random.choice(self.class_to_indices[cls]))  # 随机放回采样
                batch.append(random.choice(self.class_to_indices[cls]))  # 随机放回采样
            yield batch

    def __len__(self):
        return self.num_batches


import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

