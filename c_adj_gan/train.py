import os
import torch
from torch.utils.data import DataLoader
from gan_define import Discriminator, Generator, P, GAN

from data import MyMolDataset
from mol_adj import CH_smiles_to_adj, CH_adj_to_smiles, dense_adj_to_sparse_adj


if __name__ == '__main__':
    # 超参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    num_epoch = 120

    # 数据集
    dataset = MyMolDataset(data_file_path=os.path.join('data', 'gdb13_g.csv'),
                           transform=CH_smiles_to_adj,
                           target=['density/(g/cm3)', 'Tm/K', 'mass_calorific_value_h/(MJ/kg)', 'ISP'],
                           target_transform='01gaussian',
                           device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 模型初始化
    g = Generator(device=device).to(device)
    d = Discriminator(device=device).to(device)
    p = P(device=device).to(device)

    # 优化器
    g_optimizer = torch.optim.Adam(g.parameters(), lr=0.0001)
    d_optimizer = torch.optim.Adam(d.parameters(), lr=0.0001)
    p_optimizer = torch.optim.Adam(p.parameters(), lr=0.0001)

    # 模型组件放入GAN框架
    mol_adj_gan = GAN(d=d, g=g, p=p,
                      d_optimizer=d_optimizer, g_optimizer=g_optimizer, p_optimizer=p_optimizer,
                      dataloader=dataloader,
                      device=device)

    # 训练并保存模型和训练记录
    mol_adj_gan.train_gan(num_epoch=num_epoch)
    mol_adj_gan.plt_train_log('train_log.csv', target=['d loss', 'g loss'], save='d_g.png')
    mol_adj_gan.plt_train_log('train_log.csv', target=['d loss', 'g loss', 'p loss', 'g_p loss'], save='all.png')
    mol_adj_gan.plt_train_log('train_log.csv', target=['p loss', 'g_p loss'], save='g_g_p.png')
