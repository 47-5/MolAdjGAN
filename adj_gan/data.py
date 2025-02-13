from abc import ABC

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from mol_adj import CH_smiles_to_adj, CH_adj_to_smiles
import os


class MyMolDataset(Dataset, ABC):
    def __init__(self, data_file_path, transform=None, target=['density/(g/cm3)', 'Tm/K'], target_transform=None, device='cpu'):
        self.device = device
        self.df = pd.read_csv(data_file_path)
        self.transform = transform
        self.target_transform = target_transform
        self.df_len = len(self.df)
        self.smiles = self.df['smiles']
        self.target = target
        self.labels = np.concatenate([np.array(self.df[i]).reshape(-1, 1) for i in self.target], axis=1, dtype=np.float32)

        if transform:  # 提前都转化好，用空间换时间
            print('transform x... 将smiles转化为邻接矩阵')
            self.smiles = [np.array(transform(i), dtype=np.float32) for i in self.smiles]

        if target_transform == '01min_max':
            print('transform y... 将label压缩至[0,1]')
            min_list = np.array([min(self.df[i]) for i in self.target], dtype=np.float32)
            max_list = np.array([max(self.df[i]) for i in self.target], dtype=np.float32)
            self.labels = [(i - min_list) / (max_list - min_list) for i in self.labels]
            print('min: {}'.format(min_list))
            print('max: {}'.format(max_list))
        elif target_transform == '01gaussian':
            print('transform y... 将label转换为标准高斯分布')
            mean_list = np.array([self.df[i].mean() for i in self.target], dtype=np.float32)
            std_list = np.array([self.df[i].std() for i in self.target], dtype=np.float32)
            print('mean: {}'.format(mean_list))
            print('std: {}'.format(std_list))
            self.labels = [(i - mean_list) / std_list for i in self.labels]

        self.smiles = torch.tensor(np.array(self.smiles)).to(self.device)
        self.labels = torch.tensor(np.array(self.labels)).to(self.device)

    def __len__(self):
        return self.df_len

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        label = self.labels[idx]
        return smi, label


if __name__ == '__main__':

    dataset = MyMolDataset(data_file_path=os.path.join('data', 'remove_bad_group_and_bad_tm.csv'),
                           transform=CH_smiles_to_adj, target=['density/(g/cm3)', 'Tm/K', 'mass_calorific_value_h/(MJ/kg)', 'ISP'],
                           target_transform='01gaussian',
                           device='cpu')

    loader = DataLoader(dataset, batch_size=32)

    for idx, batch in enumerate(loader):
        x, y = batch
        print(x.shape)
        print(y.shape)
        print(y)
        break

