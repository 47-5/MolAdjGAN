from typing import Sequence
import pandas as pd
from os.path import join
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import PredictionErrorDisplay
import numpy as np
import matplotlib.pyplot as plt


def plot_r2(train_x_y_df_path=None, val_x_y_df_path=None, test_x_y_df_path=None,
            y_label_name='label', y_pre_name='pre',
            tick_number=5, tick_range_offset:Sequence[int]=None,
            ticks=None,
            save=False, save_root_path='.'):
    if tick_range_offset is None:
        tick_range_offset = [0, 0]

    # 初始化
    plt.rc('font', family='Times New Roman', size=15)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))

    text_height = 0.90


    # load data
    df_list = []
    metrics = {}
    if train_x_y_df_path is not None:
        train_df = load_df(train_x_y_df_path)
        train_df['type'] = ['train'] * len(train_df)
        train_y_label = train_df[y_label_name]
        train_y_pred = train_df[y_pre_name]
        train_r2, train_mse, train_mae, train_mape = cal_metric(train_y_label, train_y_pred, key='train',
                                                                save=save, save_root_path=save_root_path)
        metrics['train_r2'] = train_r2
        metrics['train_mse'] = train_mse
        metrics['train_mae'] = train_mae
        metrics['train_mape'] = train_mape
        df_list.append(train_df)

        PredictionErrorDisplay.from_predictions(
            train_y_label,
            y_pred=train_y_pred,
            kind="actual_vs_predicted",
            # subsample=100,
            ax=axs[0],
            random_state=0,
            scatter_kwargs={'label': 'train', 'color': 'r', 'marker': 'o', 's': 10}
        )
        PredictionErrorDisplay.from_predictions(
            train_y_label,
            y_pred=train_y_pred,
            kind="residual_vs_predicted",
            ax=axs[1],
            random_state=0,
            scatter_kwargs={'label': 'train', 'color': 'r', 'marker': 'o', 's': 10}
        )

        axs[0].text(0.05, text_height, 'train R2: {:.3f}'.format(train_r2),
                    transform=axs[0].transAxes)  # transform=plt.gca().transAxes是为了要相对于图位置的坐标，而不是数据
        axs[1].text(0.05, text_height, 'train MSE: {:.4f}'.format(train_mse),
                    transform=axs[1].transAxes)  # transform是为了要相对于图位置的坐标，而不是数据
        text_height -= 0.07

    if val_x_y_df_path is not None:
        val_df = load_df(val_x_y_df_path)
        val_df['type'] = ['val'] * len(val_df)
        val_y_label = val_df[y_label_name]
        val_y_pred = val_df[y_pre_name]
        val_r2, val_mse, val_mae, val_mape = cal_metric(val_y_label, val_y_pred, key='val',
                                                        save=save, save_root_path=save_root_path)
        metrics['val_r2'] = val_r2
        metrics['val_mse'] = val_mse
        metrics['val_mae'] = val_mae
        metrics['val_mape'] = val_mape
        df_list.append(val_df)

        PredictionErrorDisplay.from_predictions(
            val_y_label,
            y_pred=val_y_pred,
            kind="actual_vs_predicted",
            # subsample=100,
            ax=axs[0],
            random_state=0,
            scatter_kwargs={'label': 'val', 'color': 'g', 'marker': 'v', 's': 10}
        )
        PredictionErrorDisplay.from_predictions(
            val_y_label,
            y_pred=val_y_pred,
            kind="residual_vs_predicted",
            ax=axs[1],
            random_state=0,
            scatter_kwargs={'label': 'val', 'color': 'g', 'marker': 'v', 's': 10}
        )
        axs[0].text(0.05, text_height, 'val R2: {:.3f}'.format(val_r2),
                    transform=axs[0].transAxes)  # transform=plt.gca().transAxes是为了要相对于图位置的坐标，而不是数据
        axs[1].text(0.05, text_height, 'val MSE: {:.4f}'.format(val_mse),
                    transform=axs[1].transAxes)  # transform是为了要相对于图位置的坐标，而不是数据
        text_height -= 0.07

    if test_x_y_df_path is not None:
        test_df = load_df(test_x_y_df_path)
        test_df['type'] = ['test'] * len(test_df)
        test_y_label = test_df[y_label_name]
        test_y_pred = test_df[y_pre_name]
        test_r2, test_mse, test_mae, test_mape = cal_metric(test_y_label, test_y_pred, key='test',
                                                            save=save, save_root_path=save_root_path)
        metrics['test_r2'] = test_r2
        metrics['test_mse'] = test_mse
        metrics['test_mae'] = test_mae
        metrics['test_mape'] = test_mape
        df_list.append(test_df)

        PredictionErrorDisplay.from_predictions(
            test_y_label,
            y_pred=test_y_pred,
            kind="actual_vs_predicted",
            # subsample=100,
            ax=axs[0],
            random_state=0,
            scatter_kwargs={'label': 'test', 'color': 'b', 'marker': 's', 's': 10}
        )
        PredictionErrorDisplay.from_predictions(
            test_y_label,
            y_pred=test_y_pred,
            kind="residual_vs_predicted",
            ax=axs[1],
            random_state=0,
            scatter_kwargs={'label': 'test', 'color': 'b', 'marker': 's', 's': 10}
        )
        axs[0].text(0.05, text_height, 'test R2: {:.3f}'.format(test_r2),  # "{:.4f}".format(i)
                    transform=axs[0].transAxes)  # transform=plt.gca().transAxes是为了要相对于图位置的坐标，而不是数据
        axs[1].text(0.05, text_height, 'test MSE: {:.4f}'.format(test_mse),
                    transform=axs[1].transAxes)  # transform是为了要相对于图位置的坐标，而不是数据

    df = pd.concat(df_list)
    y_label = df[y_label_name]
    y_pred = df[y_pre_name]

    # cal ticks
    max_value = max(np.max(y_label), np.max(y_pred)) // 1 + 1 + tick_range_offset[0]  # //1代表向下取整
    min_value = min(np.min(y_label), np.min(y_pred)) // 1 + tick_range_offset[1]
    if ticks is None:
        ticks = np.linspace(min_value, max_value, num=tick_number)
    else:
        ticks = ticks
    # cal end

    # cal metric
    r2, mse, mae, mape = cal_metric(y_label, y_pred, key='all')
    # cal end

    axs[0].set_xticks(ticks)
    axs[0].set_yticks(ticks)
    axs[0].set_title("Actual vs. Predicted values")
    axs[0].legend()
    axs[1].set_xticks(ticks)
    axs[1].set_title("Residuals vs. Predicted Values")
    # axs[1].legend()
    # fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    if save:
        plt.savefig(join(save_root_path, 'r2.png'), dpi=1000)
    else:
        plt.show()
    return None


def load_df(path):
    if path.endswith('.xlsx'):
        df = pd.read_excel(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise NotImplemented('无法识别的文件类型!')
    return df


def cal_metric(y_label, y_pred, key='data', save=False, save_root_path='.'):
    r2 = round(r2_score(y_label, y_pred), 3)
    mse = round(mean_squared_error(y_label, y_pred), 4)
    mae = round(mean_absolute_error(y_label, y_pred), 4)
    mape = round(mean_absolute_percentage_error(y_label, y_pred), 4)
    print('-' * 50)
    print(f'{key}')
    print(f' r2: {r2} \n mse: {mse} \n mae: {mae} \n mape: {mape}')
    if save:
        with open(join(save_root_path, f'{key}.txt'), 'a') as f:
            f.write(f'{key}')
            f.write(f' r2: {r2} \n mse: {mse} \n mae: {mae} \n mape: {mape}')
    return r2, mse, mae, mape