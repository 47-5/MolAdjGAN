import numpy as np
import pandas as pd
import os
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from plot_r2 import plot_r2


def pre(p=torch.load(os.path.join('trained_model', 'p.pth')),
        df_read_path=os.path.join('data', 'gdb13_g.csv'),
        df_write_path=os.path.join('data', 'gdb_pre.csv'),
        return_real=True,):
    """把模型p的预测值输出到csv文件里，便于观察"""
    df = pd.read_csv(df_read_path)
    x = df.smiles.tolist()
    if return_real:
        properties = p.predict(smiles=x, real=True, target_transform='01gaussian').cpu().detach().numpy()
    else:
        properties = p.predict(smiles=x, real=False, target_transform='01gaussian').cpu().detach().numpy()
    print(properties.shape)
    df['pre_density'] = properties[:, 0]
    df['pre_Tm'] = properties[:, 1]
    df['pre_heat'] = properties[:, 2]
    df['pre_ISP'] = properties[:, 3]
    df.to_csv(df_write_path)
    return properties


# def plot_r2(x_y_df, y_label_name, y_pre_name, save=False):
#     plt.rc('font', family='Times New Roman', size=15)
#     if x_y_df.endswith('.xlsx'):
#         df = pd.read_excel(x_y_df)
#     elif x_y_df.endswith('.csv'):
#         df = pd.read_csv(x_y_df)
#     else:
#         print('无法识别的文件类型!')
#         return None
#     y_label = df[y_label_name]
#     y_pre = df[y_pre_name]
#     r2 = round(r2_score(y_label, y_pre), 4)
#     print('r2_score:{}'.format(r2))
#     plt.xlim(min(y_label) - 0.01 * max(y_label), max(y_label) + 0.01 * max(y_label))
#     plt.ylim(min(y_label) - 0.01 * max(y_label), max(y_label) + 0.01 * max(y_label))
#     plt.scatter(y_label, y_pre, c='royalblue', s=15)
#     plt.plot([min(y_label) - 0.01 * max(y_label), max(y_label) + 0.01 * max(y_label)],
#              [min(y_label) - 0.01 * max(y_label), max(y_label) + 0.01 * max(y_label)],
#              c='black')
#     plt.xlabel('label', size=15)
#     plt.ylabel('prediction', size=15)
#     plt.text(0.05, 0.9, 'R2_score:{}'.format(r2),
#              transform=plt.gca().transAxes)  # transform=plt.gca().transAxes是为了要相对于图位置的坐标，而不是数据
#     if save:
#         plt.savefig(save, dpi=500)
#         plt.close()
#         return None
#     plt.show()


def check_valid_uniqueness_novelty_of_g(g=torch.load(os.path.join('trained_model', 'g.pth')), number=10000,
                                        remove_error=True, remove_same=True,
                                        get_novelty=os.path.join('data', 'remove_bad_group_and_bad_tm.csv')):
    """检查生成器的valid、uniqueness、novelty"""
    print(g.generate_some_smiles(number=number, remove_error=remove_error, remove_same=remove_same,
                                 get_novelty=get_novelty))
    return None


def check_design_of_g(g=torch.load(os.path.join('trained_model', 'g.pth')), target='density', label_range=None):
    assert target in ['density', 'Tm', 'H', 'ISP'], '检查target的标签，只能取density, Tm, H, ISP'
    mean_list = torch.tensor([0.9997707, 264.84894, 41.690887, 335.81363]).to(g.device)
    std_list = torch.tensor([0.14093618, 44.425552, 1.1066712, 1.6632848]).to(g.device)
    z = torch.randn(20000, 100, 1, 1).to(g.device)

    for label_index, label_value in enumerate(label_range):
        # 生成标签
        idx = ['density', 'Tm', 'H', 'ISP'].index(target)
        label = torch.tensor([[0.9997707, 264.84894, 41.690887, 335.81363]]).to(g.device)
        label[0, idx] = label_value
        label = (label - mean_list) / std_list
        print(label)
        label = label.repeat((z.shape[0], 1)).to(g.device)

        # 生成smiles
        fake_smiles = g.generate_some_smiles(number=20000, zs=z, labels=label, remove_error=True, remove_same=False,
                                             get_novelty=False,
                                             save='{}_{}.txt'.format(target, label_index))
    return None


def generate_x_given_y(g=torch.load(os.path.join('trained_model', 'g.pth')), y=[1.100, 220.0, 43.00, 337.0], number=20000):
    mean_list = torch.tensor([0.9997707, 264.84894, 41.690887, 335.81363]).to(g.device)
    std_list = torch.tensor([0.14093618, 44.425552, 1.1066712, 1.6632848]).to(g.device)
    z = torch.randn(number, 100, 1, 1).to(g.device)
    label = torch.tensor([y]).to(g.device)
    label = (label - mean_list) / std_list
    print(label)
    label = label.repeat((z.shape[0], 1)).to(g.device)

    # 生成smiles
    fake_smiles = g.generate_some_smiles(number=20000, zs=z, labels=label, remove_error=True, remove_same=False,
                                         get_novelty=False,
                                         save='design.txt')
    return None


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
    plt.plot(range(1, len(result) + 1), np.array(result) / 32)
    plt.xlabel('Epoch')
    plt.ylabel('valid percentage')
    if save:
        plt.savefig(save, dpi=600)
        plt.close()
    else:
        plt.show()
    return None


def find_classical_fuel_molecules(target_smi, target_properties, try_number=20000,
                                  g=torch.load(os.path.join('trained_model', 'g.pth'))):
    label = torch.tensor([target_properties]).to(g.device).repeat((try_number, 1))
    fake_smiles = g.generate_some_smiles(number=try_number, zs=None, labels=label, remove_error=True, remove_same=False,
                                         get_novelty=False,
                                         save='find_{}.txt'.format(target_smi))
    for i in fake_smiles:
        if i == target_smi:
            print('find it!')
            return True
    return False


if __name__ == '__main__':

    # 加载模型
    g = torch.load(os.path.join('trained_model', 'g.pth'))
    p = torch.load(os.path.join('trained_model', 'p.pth'))

    # # 预测燃料性质
    # pre(p=p,
    #     df_read_path=os.path.join('data', 'gdb13_g.csv'),
    #     df_write_path=os.path.join('data', 'gdb13_g_pre.csv'),
    #     return_real=True, )
    # pre(p=p,
    #     df_read_path=os.path.join('data', 'gdb13_g_train.csv'),
    #     df_write_path=os.path.join('data', 'gdb13_g_train_pre.csv'),
    #     return_real=True, )
    # pre(p=p,
    #     df_read_path=os.path.join('data', 'gdb13_g_val.csv'),
    #     df_write_path=os.path.join('data', 'gdb13_g_val_pre.csv'),
    #     return_real=True, )
    # pre(p=p,
    #     df_read_path=os.path.join('data', 'gdb13_g_test.csv'),
    #     df_write_path=os.path.join('data', 'gdb13_g_test_pre.csv'),
    #     return_real=True,)

    # 画图
    # plot_r2(train_x_y_df_path=os.path.join('data', 'gdb13_g_train_pre.csv'),
    #         val_x_y_df_path=os.path.join('data', 'gdb13_g_val_pre.csv'),
    #         test_x_y_df_path=os.path.join('data', 'gdb13_g_test_pre.csv'),
    #         y_label_name='density/(g/cm3)',
    #         y_pre_name='pre_density',
    #         tick_number=None, tick_range_offset=None,
    #         ticks=[0.5, 0.75, 1.0, 1.25, 1.5],
    #         save=True, save_root_path='.'
    #         )
    # plot_r2(train_x_y_df_path=os.path.join('data', 'gdb13_g_train_pre.csv'),
    #         val_x_y_df_path=os.path.join('data', 'gdb13_g_val_pre.csv'),
    #         test_x_y_df_path=os.path.join('data', 'gdb13_g_test_pre.csv'),
    #         y_label_name='Tm/K',
    #         y_pre_name='pre_Tm',
    #         tick_number=None, tick_range_offset=None,
    #         ticks=[100, 150, 200, 250, 300],
    #         save=True, save_root_path='.'
    #         )
    # plot_r2(train_x_y_df_path=os.path.join('data', 'gdb13_g_train_pre.csv'),
    #         val_x_y_df_path=os.path.join('data', 'gdb13_g_val_pre.csv'),
    #         test_x_y_df_path=os.path.join('data', 'gdb13_g_test_pre.csv'),
    #         y_label_name='mass_calorific_value_h/(MJ/kg)',
    #         y_pre_name='pre_heat',
    #         tick_number=None, tick_range_offset=None,
    #         ticks=[40, 42, 44, 46],
    #         save=True, save_root_path='.'
    #         )
    # plot_r2(train_x_y_df_path=os.path.join('data', 'gdb13_g_train_pre.csv'),
    #         val_x_y_df_path=os.path.join('data', 'gdb13_g_val_pre.csv'),
    #         test_x_y_df_path=os.path.join('data', 'gdb13_g_test_pre.csv'),
    #         y_label_name='ISP',
    #         y_pre_name='pre_ISP',
    #         tick_number=None, tick_range_offset=None,
    #         ticks=[335, 336, 337, 338, 339],
    #         save=True, save_root_path='.'
    #         )

    # # 检查生成器的valid、uniqueness、novelty
    # check_valid_uniqueness_novelty_of_g(g=g)

    # 检查设计
    # check_design_of_g(g=g, target='density', label_range=[0.90, 1.00, 1.10])
    # check_design_of_g(g=g, target='Tm', label_range=[220, 260, 300])
    # check_design_of_g(g=g, target='H', label_range=[40.0, 41.5, 43.0])
    # check_design_of_g(g=g, target='ISP', label_range=[335.0, 336.0, 337.0])

    # 设计具有理想性质的分子
    # generate_x_given_y(g=g, y=[1.100, 220.0, 43.00, 337.0], number=20000)

    # # 绘制R2
    # plot_r2(x_y_df='with_pre.csv', y_label_name='density/(g/cm3)', y_pre_name='pre_density', save='c_gan_density')
    # plot_r2(x_y_df='with_pre.csv', y_label_name='Tm/K', y_pre_name='pre_Tm', save='c_gan_Tm')
    # plot_r2(x_y_df='with_pre.csv', y_label_name='mass_calorific_value_h/(MJ/kg)', y_pre_name='pre_heat', save='c_gan_heat')
    # plot_r2(x_y_df='with_pre.csv', y_label_name='ISP', y_pre_name='pre_ISP', save='pre_ISP')

    # # 绘制训练过程中生成器的生成质量
    # plot_generate_quality(dir=os.path.join('fake'), save='valid_percentage_in_training_process.png')

    # # 找经典燃料分子
    # stop = False
    # while not stop:
    #     find = find_classical_fuel_molecules(target_smi='C1CC2C3CCC(C3)C2C1',
    #                                          target_properties=[1.044, 269.32, 42.088, 337.037],
    #                                          try_number=30000,
    #                                          g=g)
    #     print(find)
    #     if find:
    #         stop = True
