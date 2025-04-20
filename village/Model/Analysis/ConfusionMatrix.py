import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets
from torch.utils.data import DataLoader
import Config.pytorch as pytorch_model
from Config.get_config import load_config

warnings.filterwarnings('ignore')


def test(model, device, test_loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1, keepdim=True)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


if __name__ == '__main__':
    # 填配置文件里的模型的键名:
    config_model_name = 'Improve_ConvNeXt'

    config = load_config()

    # 测试集所在的路径
    test_root = config['project_path'] + '/' + config['test_data_path']

    # 需要测试的模型所在的路径
    model_path = config['project_path'] + '/' + config[config_model_name]

    # 不同类别的名称列表
    names = config['display_labels']
    display_labels = [str(i) for i in range(len(names))]

    # 批次大小
    batch_size = config['batch_size']

    # 是否要在混淆矩阵每个单元格上显示具体数值
    show_figure = config['show_figure']

    # 是否要对结果进行归一化
    normalization = config['normalization']

    # 使用什么机器训练模型
    device = config['device']

    # 类别数量
    num_classes = config['num_classes']

    plt.rcParams['font.sans-serif'] = ['SimHei']
    img_size = 224
    transform = pytorch_model.val_data(img_size)

    # 加载测试数据
    test_dataset = datasets.ImageFolder(test_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = torch.load(model_path, map_location=torch.device(device)).to(device)

    # 预测
    y_pred, y_true = test(model, device, test_loader)

    if normalization:
        # 归一化
        cm = confusion_matrix(y_true, y_pred, normalize='true')
    else:
        # 不进行归一化
        cm = confusion_matrix(y_true, y_pred)

    # 打印混淆矩阵
    print("Confusion Matrix: ")
    for i in range(0, len(cm)):
        print(cm[i])
        print("____________________________________________")

    # 调整图形尺寸
    fig, ax = plt.subplots(figsize=(14, 8))  # 增加宽度

    # 画出混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(
        include_values=show_figure,
        cmap="Blues",
        ax=ax,
        xticks_rotation="horizontal"
    )

    # 添加图注解释
    label_explanation = {i: names[i] for i in range(len(names))}

    # 创建一个新的轴，用于放置图注
    legend_ax = fig.add_axes([0.1, 0.1, 0.3, 0.8])  # [left, bottom, width, height]
    legend_ax.axis('off')  # 隐藏新轴

    # 在新轴中添加图注
    legend_ax.text(
        0, 0.5, '\n'.join([f"{k}: {v}" for k, v in label_explanation.items()]),
        verticalalignment='center',
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5)
    )

    now = int(time.time()) * 25
    save_dir = './result/ConfusionMatrix_' + config_model_name + '_' + str(now) + '.png'
    plt.savefig(save_dir, dpi=500, bbox_inches='tight')
    plt.show()
