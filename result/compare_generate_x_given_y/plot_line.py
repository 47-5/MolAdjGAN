import pandas as pd
import numpy as np
from typing import Sequence
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 首先配置字体信息
config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)


def plot_line(df_path, specie_data_label: str = 'FP_type', bar_data_label: Sequence[str] = None,
              bar_plot_label: Sequence[str] = None,
              x_axis_label=None, xtick_label=None,
              y_axis_label=None, ytick_label=None, y_lim=None,
              save=False, save_path='bar.png'):
    if bar_plot_label is None:
        bar_plot_label = bar_data_label
    if df_path.endswith('xlsx'):
        df = pd.read_excel(df_path)
    elif df_path.endswith('csv'):
        df = pd.read_csv(df_path)
    else:
        raise NotImplementedError('df_path must be xlsx or csv')
    dict_columns = df.to_dict('list')  # 每个列名作为键，值为列表
    print(dict_columns)

    # 整理数据，做一些基本设置
    species = dict_columns[specie_data_label]
    bar_data = {k: dict_columns[k] for k in bar_data_label}
    n_var = len(species)
    x = np.arange(n_var)

    # 初始化布局
    fig, ax = plt.subplots(layout='constrained')
    # 添加多个柱子
    for index, (attribute, measurement) in enumerate(bar_data.items()):
        ax.scatter(x=x, y=measurement)
        ax.plot(x, measurement, label=bar_plot_label[index])



    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    # ax.set_title('Title')
    if xtick_label is not None:
        ax.set_xticks(x, [f'{i:.2f}' for i in xtick_label])  # todo
    if ytick_label is not None:
        ax.set_yticks(ytick_label)
    ax.xaxis.set_tick_params(direction='in')  # 刻度线朝内
    ax.yaxis.set_tick_params(direction='in')
    # ax.legend(loc='upper left', ncols=len(bar_data_label), labels=bar_plot_label)
    ax.legend()
    ax.set_ylim(y_lim)

    if save:
        plt.savefig(save_path, dpi=1000)
        plt.close()
    else:
        plt.show()
        plt.close()
    return None


if __name__ == '__main__':


    # df_path = 'density.xlsx'
    # specie_data_label = 'target'
    # bar_data_label = ['c_info_mean', 'c_mean', 'target']
    # bar_plot_label = ['c-infoGAN', 'cGAN', 'target']
    # x_axis_label = 'Target'
    # xtick_label = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    # y_axis_label = 'Mean density of the designed molecules'  # 若不写\mathrm{}则会是斜体的效果
    # ytick_label = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    # y_lim = None
    # save = True
    # save_path = 'density.png'
    #
    # plot_line(df_path=df_path, specie_data_label=specie_data_label, bar_data_label=bar_data_label,
    #           bar_plot_label=bar_plot_label, x_axis_label=x_axis_label, xtick_label=xtick_label,
    #           y_axis_label=y_axis_label, ytick_label=ytick_label, y_lim=y_lim, save=save, save_path=save_path)

    # df_path = 'H.xlsx'
    # specie_data_label = 'target'
    # bar_data_label = ['c_info_mean', 'c_mean', 'target']
    # bar_plot_label = ['c-infoGAN', 'cGAN', 'target']
    # x_axis_label = 'Target'
    # xtick_label = [42, 42.25, 42.5, 42.75, 43]
    # y_axis_label = 'Mean calorific value of the designed molecules'  # 若不写\mathrm{}则会是斜体的效果
    # ytick_label = [42, 42.25, 42.5, 42.75, 43]
    # y_lim = None
    # save = True
    # save_path = 'H.png'
    #
    # plot_line(df_path=df_path, specie_data_label=specie_data_label, bar_data_label=bar_data_label,
    #           bar_plot_label=bar_plot_label, x_axis_label=x_axis_label, xtick_label=xtick_label,
    #           y_axis_label=y_axis_label, ytick_label=ytick_label, y_lim=y_lim, save=save, save_path=save_path)

    # df_path = 'Tm.xlsx'
    # specie_data_label = 'target'
    # bar_data_label = ['c_info_mean', 'c_mean', 'target']
    # bar_plot_label = ['c-infoGAN', 'cGAN', 'target']
    # x_axis_label = 'Target'
    # xtick_label = [230, 240, 250, 260, 270, 280]
    # y_axis_label = 'Mean freezing point of the designed molecules'  # 若不写\mathrm{}则会是斜体的效果
    # ytick_label = [230, 240, 250, 260, 270, 280]
    # y_lim = None
    # save = True
    # save_path = 'Tm.png'
    #
    # plot_line(df_path=df_path, specie_data_label=specie_data_label, bar_data_label=bar_data_label,
    #           bar_plot_label=bar_plot_label, x_axis_label=x_axis_label, xtick_label=xtick_label,
    #           y_axis_label=y_axis_label, ytick_label=ytick_label, y_lim=y_lim, save=save, save_path=save_path)

    df_path = 'ISP.xlsx'
    specie_data_label = 'target'
    bar_data_label = ['c_info_mean', 'c_mean', 'target']
    bar_plot_label = ['c-infoGAN', 'cGAN', 'target']
    x_axis_label = 'Target'
    xtick_label = [336.75, 337, 337.25, 337.5, 337.75]
    y_axis_label = 'Mean specific impulse of the designed molecules'  # 若不写\mathrm{}则会是斜体的效果
    ytick_label = [336.75, 337, 337.25, 337.5, 337.75]
    y_lim = None
    save = True
    save_path = 'ISP.png'

    plot_line(df_path=df_path, specie_data_label=specie_data_label, bar_data_label=bar_data_label,
              bar_plot_label=bar_plot_label, x_axis_label=x_axis_label, xtick_label=xtick_label,
              y_axis_label=y_axis_label, ytick_label=ytick_label, y_lim=y_lim, save=save, save_path=save_path)