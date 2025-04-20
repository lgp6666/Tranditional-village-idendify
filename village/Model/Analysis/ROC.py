import numpy as np
import time
from sklearn.metrics import roc_curve, auc
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random
import Config.pytorch as pytorch_model
from Config.get_config import load_config
import warnings
warnings.filterwarnings('ignore')


def generate_distinguishable_colors(num_colors):
    colors = []
    while len(colors) < num_colors:
        color = random.choice(list(mcolors.TABLEAU_COLORS.values()))  # Use TABLEAU palette for better visibility
        if mcolors.rgb_to_hsv(mcolors.to_rgb(color))[2] < 0.9:  # Filter out very light colors
            colors.append(color)
    return colors


def test(model, device, test_loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


if __name__ == '__main__':
    # 填配置文件里的模型的键名:
    config_model_name = 'Improve_ConvNeXt'

    config = load_config()

    # 不同类别的名称列表
    labels = config['display_labels']

    # 测试集所在的路径
    test_dir = config['project_path'] + '/' + config['test_data_path']

    # 需要测试的模型所在的路径
    model_path = config['project_path'] + '/' + config[config_model_name]

    # 训练使用的机器
    device = config['device']

    # 批次大小
    batch_size = config['batch_size']

    # 类别数量
    num_classes = config['num_classes']

    img_size = 224
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    # 加载测试数据
    test_dataset = datasets.ImageFolder(test_dir, transform=pytorch_model.val_data(img_size))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = torch.load(model_path, map_location=torch.device(device)).to(device)

    # 预测
    y_pred, y_true = test(model, device, test_loader)

    # 将真实标签转化为one-hot编码
    num_classes = len(labels)
    y_true = np.eye(num_classes)[y_true]

    # 计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线
    fig, ax = plt.subplots(figsize=(10, 10))  # 调整图形尺寸
    lw = 2
    colors = generate_distinguishable_colors(num_classes)
    for i, color in zip(range(num_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='{0}'.format(labels[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

    now = int(time.time()) * 25
    save_dir = './result/ROC_' + config_model_name + '_' + str(now) + '.png'
    fig.savefig(save_dir, dpi=500, bbox_inches='tight')
    plt.show()
