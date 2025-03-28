import os

import numpy as np
import torch
from rdkit import Chem
import matplotlib.pyplot as plt


def check_valid_uniqueness_novelty_of_g(g=torch.load(os.path.join('trained_model', 'g.pth')), number=10000, remove_error=True, remove_same=True,
                                        get_novelty=os.path.join('data', 'remove_bad_group_and_bad_tm.csv')):
    """检查生成器的valid、uniqueness、novelty"""
    print(g.generate_some_smiles(number=number, remove_error=remove_error, remove_same=remove_same,
                                 get_novelty=get_novelty))
    return None


def get_interpolation(z1, z2, point_number=10):
    """插值"""
    z_list = [i * z1 + (1 - i) * z2 for i in torch.linspace(start=0, end=1, steps=point_number)]
    return torch.cat(z_list, dim=0)


def molecular_linear_algebra_1(g=torch.load(os.path.join('trained_model', 'g.pth'))):
    z1 = torch.randn(1, 100, 1, 1)
    z2 = torch.randn(1, 100, 1, 1)
    z3 = (z1 + z2) / 2
    zs = torch.cat([z1, z2, z3], dim=0).to(g.device)
    fake_smiles = g.generate_some_smiles(zs=zs, number=3, remove_error=True, remove_same=False, get_novelty=False,)
    return fake_smiles


def molecular_linear_algebra_2(g=torch.load(os.path.join('trained_model', 'g.pth'))):
    z1 = torch.randn(1, 100, 1, 1)
    z2 = torch.randn(1, 100, 1, 1)
    z3 = torch.randn(1, 100, 1, 1)
    z4 = z1 - z2 + z3
    zs = torch.cat([z1, z2, z3, z4], dim=0).to(g.device)
    fake_smiles = g.generate_some_smiles(zs=zs, number=4, remove_error=True, remove_same=False, get_novelty=False,)
    return fake_smiles


def plot_generate_quality(dir=os.path.join('fake'), save=None):
    file_names = sorted(os.listdir(dir), key=lambda i: int(i.split('.')[0]))
    result = []
    for file in file_names:
        valid_number = 0
        f = open(os.path.join(dir, file), 'r')
        for smi in f.readlines():
            if smi.startswith('error'):
                pass
            else:
                valid_number += 1
        result.append(valid_number)
    plt.plot(range(1, len(result)+1), np.array(result)/32)
    plt.xlabel('Epoch')
    plt.ylabel('valid percentage')
    if save:
        plt.savefig(save, dpi=600)
        plt.close()
    else:
        plt.show()
    return None


if __name__ == '__main__':

    # 加载模型
    g = torch.load(os.path.join('trained_model', 'g.pth'))

    # # 检查生成器的valid、uniqueness、novelty
    check_valid_uniqueness_novelty_of_g(g=g)

    # # 线性插值
    # z1 = torch.randn(1, 100, 1, 1)
    # z2 = torch.randn(1, 100, 1, 1)
    # zs = get_interpolation(z1, z2, 10)
    # fake_smiles = g.generate_some_smiles(zs, number=10, count_error=True, remove_same=False, get_novelty=False,
    #                                      save='interpolation6.txt')
    # print(fake_smiles)

    # # 分子线性代数
    # linear_algebra = molecular_linear_algebra_1(g=g)
    # print(linear_algebra)

    # linear_algebra = molecular_linear_algebra_2(g=g)
    # print(linear_algebra)

    # # 绘制训练过程中生成器的生成质量
    # plot_generate_quality(dir=os.path.join('fake'), save='valid_percentage_in_training_process.png')

