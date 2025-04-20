import os.path
import pandas as pd
import matplotlib.pyplot as plt
from Config.get_config import load_config


def compare_loss_train(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Train Loss'], label="Train_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Loss")
    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("Loss Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Train_Loss_compare.png"), dpi=300, bbox_inches='tight')


def compare_loss_val(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Valid Loss'], label="Valid_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Loss")
    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("Loss Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Val_Loss_compare.png"), dpi=300, bbox_inches='tight')


def compare_acc_train(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Train Accuracy'], label="Train_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("Accuracy Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Train_Accuracy_compare.png"), dpi=300, bbox_inches='tight')

def compare_acc_val(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Valid Accuracy'], label="Valid_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')

    # 表格标题
    plt.title("Accuracy Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Val_Accuracy_compare.png"), dpi=300, bbox_inches='tight')


def compare_precision_train(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Train Precision'], label="Train_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Precision")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("Precision Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Train_Precision_compare.png"), dpi=300, bbox_inches='tight')


def compare_precision_val(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Valid Precision'], label="Valid_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Precision")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("Precision Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Val_Precision_compare.png"), dpi=300, bbox_inches='tight')


def compare_recall_train(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Train Recall'], label="Train_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Recall")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("Recall Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Train_Recall_compare.png"), dpi=300, bbox_inches='tight')


def compare_recall_val(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Valid Recall'], label="Valid_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("Recall")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("Recall Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Val_Recall_compare.png"), dpi=300, bbox_inches='tight')


def compare_f1_train(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Train F1'], label="Train_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("F1_Score")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("F1_Score Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Train_F1_Score_compare.png"), dpi=300, bbox_inches='tight')


def compare_f1_val(my_csvs):
    plt.figure(figsize=(12, 8))  # 调整图形尺寸
    for one_csv in my_csvs:
        one_path = one_csv['path']
        one_name = one_csv['name']
        one_results = pd.read_csv(one_path)
        try:
            plt.plot(one_results['Valid F1'], label="Valid_" + one_name)
        except Exception as e:
            print('检查文件：' + one_path)

    # 横坐标表示为
    plt.xlabel("Epoch")
    # 纵坐标表示为
    plt.ylabel("F1_Score")
    plt.ylim(0, 1)  # 设置纵坐标的范围

    # 创建一个新的轴用于放置图注
    legend_ax = plt.gca().inset_axes([1.05, 0, 0.2, 1])
    legend_ax.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc='upper right')
    # 表格标题
    plt.title("F1_Score Comparison")
    # 保存路径
    plt.savefig(os.path.join(r'./', "Val_F1_Score_compare.png"), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    config = load_config()
    base_path = config['project_path']
    my_csvs = [
        {'name': 'AlexNet',
         'path': 'AlexNet_43350754000_metrics.csv'},

        {'name': 'ConvNeXt',
         'path': 'ConvNeXt_tiny_43350766800_metrics.csv'},

        {'name': 'DenseNet',
         'path': 'DenseNet121_43350789250_metrics.csv'},

        {'name': 'EfficientNet',
         'path': 'EfficientNetV2_S_43350849800_metrics.csv'},

        {'name': 'GoogLeNet',
         'path': 'GoogLeNet_43350888225_metrics.csv'},

        {'name': 'Improve_ConvNeXt',
         'path': 'Improve_ConvNeXt_43350726900_metrics.csv'},

        {'name': 'MobileNet',
         'path': 'MobileNet_V3_Small_43350909775_metrics.csv'},

        {'name': 'RegNet',
         'path': 'RegNetX_400MF_43350926850_metrics.csv'},

        {'name': 'ResNet',
         'path': 'ResNet50_43350960550_metrics.csv'},

        {'name': 'ShuffleNet',
         'path': 'ShuffleNet_V2_x1_0_43351008250_metrics.csv'},

        {'name': 'Swin_Transformer',
         'path': 'swin_tiny_patch4_window7_224_43351031550_metrics.csv'},

        {'name': 'VGG',
         'path': 'VGG16_43351058125_metrics.csv'},
    ]
    compare_loss_train(my_csvs)
    compare_loss_val(my_csvs)
    compare_acc_train(my_csvs)
    compare_acc_val(my_csvs)
    compare_recall_train(my_csvs)

    compare_precision_train(my_csvs)
    compare_precision_val(my_csvs)
    compare_f1_train(my_csvs)
    compare_f1_val(my_csvs)
    compare_recall_val(my_csvs)

