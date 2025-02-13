import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from mol_adj import CH_smiles_to_adj, CH_adj_to_smiles, dense_adj_to_sparse_adj


class Generator(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Generator, self).__init__()
        self.device = device

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=24 * 8, kernel_size=(4, 4), stride=(1, 1),
                               padding=(0, 0), bias=False),
            nn.BatchNorm2d(24 * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=24 * 8, out_channels=24 * 4, kernel_size=(4, 4), stride=(1, 1),
                               padding=(0, 0), bias=False),
            nn.BatchNorm2d(24 * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=24 * 4, out_channels=24 * 2, kernel_size=(4, 4), stride=(1, 1),
                               padding=(0, 0), bias=False),
            nn.BatchNorm2d(24 * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=24 * 2, out_channels=24 * 1, kernel_size=(4, 4), stride=(1, 1),
                               padding=(0, 0), bias=False),
            nn.BatchNorm2d(24 * 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=24 * 1, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False),
            nn.Sigmoid()
        )

    def forward(self, z):
        adj = self.net(z)
        return (adj + adj.permute(0, 1, 3, 2)) / 2

    def generate_some_smiles(self, number=50, zs=None, remove_error=True, remove_same=True, get_novelty=False, save=None):
        """
        生成smiles
        :param number:生成数量，如果给定zs和label都给定则无效
        :param zs: 随机隐变量z
        :param labels: 标签
        :param remove_error: 是否统计错误数并且剔除无效smiles
        :param remove_same: 是否统计不重复的smiles数目并且只保留不重复的smiles
        :param get_novelty: 是否统计没有出现在训练集中的分子数目并且只保留没出现在训练集中的smiles
        :param save: 是否保存生成的路径，如果保存，应该设置为保存路径，否则为None，默认值为None
        :return: 生成的smiles列表
        """

        if zs is None:
            zs = torch.randn(number, 100, 1, 1).to(self.device)

        fake_adj_s = self.forward(zs)
        fake_smiles = [CH_adj_to_smiles(dense_adj_to_sparse_adj(i[0].detach().cpu().numpy())) for i in fake_adj_s]
        if remove_error:
            error_smiles_number = fake_smiles.count('error smiles')
            print(
                'error smiles number:{}  valid:{}'.format(error_smiles_number, (number - error_smiles_number) / number))
            fake_smiles = [i for i in fake_smiles if i != 'error smiles']
        if remove_same:
            fake_smiles = list(set(fake_smiles))
            uniqueness_smiles_number = len(fake_smiles)
            print('不重复的smiles共{}个  uniqueness:{}'.format(uniqueness_smiles_number, uniqueness_smiles_number / number))
        if get_novelty:
            smiles = pd.read_csv(get_novelty)['smiles'].tolist()
            fake_smiles = [i for i in fake_smiles if i not in smiles]
            novelty_smiles_number = len(fake_smiles)
            print('不在训练集里的smiles共{}  novelty:{}'.format(novelty_smiles_number, novelty_smiles_number / number))
        if save is not None:
            with open(save, 'w') as f:
                for i in fake_smiles:
                    f.write(i + '\n')
        return fake_smiles


class Discriminator(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Discriminator, self).__init__()
        self.device = device
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=1, out_channels=24 * 4, kernel_size=(13, 13), stride=(1, 1), padding=(0, 0),
                          bias=False)),
            nn.BatchNorm2d(24 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(in_channels=24 * 4, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                          bias=False))
        )

    def forward(self, adj):
        p = self.net(adj)
        return p.reshape(-1, 1)


class GAN(nn.Module):
    def __init__(self, d, g, d_optimizer, g_optimizer, dataloader,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(GAN, self).__init__()
        # 模型和对应的优化器
        self.d = d
        self.g = g
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        # dataloader和device
        self.dataloader = dataloader
        self.device = device

    def train_gan(self, num_epoch, ):
        print('use:{}'.format(self.device))

        # 损失函数和标签
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        one_label = torch.ones(self.dataloader.batch_size, 1).to(self.device)
        zero_label = torch.zeros(self.dataloader.batch_size, 1).to(self.device)

        # 空字典，用来记录每个epoch的损失，从而可以绘制训练过程损失函数曲线
        log = {
            'd loss': [],
            'g loss': [],
        }

        # 训练主循环
        for epoch in range(1, num_epoch + 1):
            print('-' * 60)
            print('epoch {} / {}'.format(epoch, num_epoch))
            d_epoch_loss = 0.0
            g_epoch_loss = 0.0
            step = 0

            # 训练内循环，即在dataloader上循环一边，即为1个epoch
            for index, batch in enumerate(self.dataloader):
                step += 1

                # 取真实样本和标签
                real_sample, labels = batch
                real_sample = real_sample.to(self.device).reshape(-1, 1, 13, 13)
                labels = labels.to(self.device)

                # 生成假样本
                z = torch.randn(self.dataloader.batch_size, 100, 1, 1).to(self.device)
                fake_sample = self.g(z)

                # 更新d的参数
                self.d_optimizer.zero_grad()
                d_loss = bce(self.d(real_sample), one_label) + \
                         bce(self.d(fake_sample.detach()), zero_label)  # 这里.detach()是因为训练判别器时不需要再对这个记录梯度了
                d_loss.backward()
                self.d_optimizer.step()

                # 更新g的参数
                self.g_optimizer.zero_grad()
                g_loss = bce(self.d(fake_sample), one_label)
                g_loss.backward()
                self.g_optimizer.step()

                # 更新损失记录
                d_epoch_loss += d_loss.item()
                g_epoch_loss += g_loss.item()
                # 结束内循环

            # 更新log
            d_epoch_loss /= step
            g_epoch_loss /= step
            log['d loss'].append(d_epoch_loss)
            log['g loss'].append(g_epoch_loss)

            # 打印训练过程中每个epoch的损失
            print('d loss {}'.format(d_epoch_loss))
            print('g loss {}'.format(g_epoch_loss))

            # todo 这里是想要在每个epoch结束后保存这个epoch最后一个batch生成器g生成的样本以观察训练情况。 如果用别的数据训练这里必须修改
            if not os.path.exists(os.path.join('fake')):
                os.mkdir(os.path.join('fake'))
            with open(os.path.join('fake', '{}.txt'.format(epoch)).format(epoch), 'w') as f:
                for fake_sample_index, one_fake_sample in enumerate(fake_sample):
                    one_fake_sample = one_fake_sample[0].detach().cpu().numpy()  #
                    one_fake_sample = dense_adj_to_sparse_adj(one_fake_sample)
                    smi = CH_adj_to_smiles(one_fake_sample)
                    f.write(smi + '\n')

        # 训练主循环结束后把记录损失函数变化情况的log转化成csv文件输入
        log_df = pd.DataFrame(log, index=[i for i in range(1, num_epoch + 1)])
        log_df.to_csv(path_or_buf=r'train_log.csv', index_label='epoch')

        # 保存模型
        if not os.path.exists(os.path.join('trained_model')):
            os.mkdir(os.path.join('trained_model'))
        torch.save(self.g, os.path.join('trained_model', 'g.pth'))
        torch.save(self.d, os.path.join('trained_model', 'd.pth'))
        return log_df

    @ staticmethod
    def plt_train_log(log_csv_path, target=[], save=False):
        df = pd.read_csv(log_csv_path)
        epochs = df['epoch'].tolist()
        for i in target:
            plt.plot(epochs, df[i], label=i)
        plt.legend()

        if epochs[-1] > 10:
            plt.xticks(ticks=[i for i in range(epochs[0], epochs[-1], int(epochs[-1] / 10))])
        else:
            plt.xticks(ticks=[i for i in epochs])

        if save:
            plt.savefig(save, )
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':

    print('这个py文件里定义了GAN的基类模型')

    from torchviz import make_dot

    g = Generator()
    d = Discriminator()

    # x = torch.randn(32, 1, 13, 13)
    # y = d(x)
    # d_figure = make_dot(y)
    # d_figure.view()

    # z = torch.randn(1, 100, 1, 1)
    # x_hat = g(z)
    # g_figure = make_dot(x_hat)
    # g_figure.view()




