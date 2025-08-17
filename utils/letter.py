import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
import sys
# from imblearn.over_sampling import SMOTE
# sys.path.insert(0, "./")

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

def min_max_scaling(df):
    df_norm = df.copy()
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
    return df_norm

class LetterDataset(data.Dataset):
    def __init__(self, transform, train=True):
        """
        Args:
            csv_path (string): Path to the csv file.
        """
        csv_path = "/home/shunjie/codes/defend_label_inference/cs/Datasets/letter/letter-recognition.data"
        
        self.train = train
        self.df = pd.read_csv(csv_path)

        self.df.columns = ['Label'] + [f'f{i}' for i in range(1, self.df.shape[1])]
        
        le = LabelEncoder()
        self.df['Label'] = le.fit_transform(self.df['Label'])

        y = self.df["Label"].values  # 提取标签
        x = self.df.drop(columns=["Label"])  # 去除标签列
        x = min_max_scaling(x)  # 对特征归一化

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

        sc = StandardScaler()

        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)

        self.train_data = x_train  # numpy array
        self.test_data = x_test

        self.train_labels = y_train.tolist()
        self.test_labels = y_test.tolist()

        print(csv_path, "train", len(self.train_data), "test", len(self.test_data))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            data, label = self.train_data[index], self.train_labels[index]
        else:
            data, label = self.test_data[index], self.test_labels[index]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
