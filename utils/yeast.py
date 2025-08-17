import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
import sys
# from imblearn.over_sampling import SMOTE
# sys.path.insert(0, "./")
from collections import Counter

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
# from datasets.dataset_setup import DatasetSetup
# from my_utils.utils import train_val_split

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns


# class YeastDataset(data.Dataset):

#     def __init__(self, transform, train=True):
#         """
#         Args:
#             csv_path (string): Path to the csv file.
#         """
#         csv_path = "/home/shunjie/codes/defend_label_inference/cs/Datasets/yeast/yeast.data"
        
#         self.train = train
#         self.df = pd.read_csv(csv_path, sep=r'\s+')

#         self.df.columns = ['Name', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'Label']
#         self.df = self.df.drop(columns=['Name'])
#         # # 假设label是最后一列
#         le = LabelEncoder()
#         self.df['Label'] = le.fit_transform(self.df['Label'])

#         y = self.df["Label"].values  # 标签
#         x = self.df.drop(columns=["Label"]).values  # 特征（去除label列）

#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

#         sc = StandardScaler()

#         x_train = sc.fit_transform(x_train)
#         x_test = sc.fit_transform(x_test)

#         self.train_data = x_train  # numpy array
#         self.test_data = x_test

#         self.train_labels = y_train.tolist()
#         self.test_labels = y_test.tolist()

#         print(csv_path, "train", len(self.train_data), "test", len(self.test_data))

#     def __len__(self):
#         if self.train:
#             return len(self.train_data)
#         else:
#             return len(self.test_data)

#     def __getitem__(self, index):
#         if self.train:
#             data, label = self.train_data[index], self.train_labels[index]
#         else:
#             data, label = self.test_data[index], self.test_labels[index]

#         return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)



class YeastDataset(data.Dataset):
    def __init__(self, transform=None, train=True):
        csv_path = "/home/shunjie/codes/defend_label_inference/cs/Datasets/yeast/yeast.data"

        self.train = train
        self.df = pd.read_csv(csv_path, sep=r'\s+')
        self.df.columns = ['Name', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'Label']
        self.df = self.df.drop(columns=['Name'])

        # 标签编码
        le = LabelEncoder()
        self.df['Label'] = le.fit_transform(self.df['Label'])

        # 统计标签分布
        label_counts = Counter(self.df['Label'])
        valid_labels = [label for label, count in label_counts.items() if count >= 100]
        # print(f"[YeastDataset] Kept labels (≥100 samples): {valid_labels}")

        # 过滤样本
        self.df = self.df[self.df['Label'].isin(valid_labels)].reset_index(drop=True)

        # 重新编码标签（从0开始）
        new_label_encoder = LabelEncoder()
        self.df['Label'] = new_label_encoder.fit_transform(self.df['Label'])

        # 特征和标签
        y = self.df["Label"].values
        x = self.df.drop(columns=["Label"]).values

        # 划分训练和测试
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16, stratify=y)

        # 标准化
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        self.train_data = x_train
        self.test_data = x_test
        self.train_labels = y_train.tolist()
        self.test_labels = y_test.tolist()

        # print(csv_path, "train", len(self.train_data), "test", len(self.test_data))

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, index):
        data = self.train_data[index] if self.train else self.test_data[index]
        label = self.train_labels[index] if self.train else self.test_labels[index]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)