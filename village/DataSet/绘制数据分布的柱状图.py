import os
import matplotlib.pyplot as plt
import random

if __name__ == '__main__':
    # 文件夹路径
    folder_path = 'ori'

    # 获取每个类别的图片数量
    category_counts = {}
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            image_count = len(
                [img for img in os.listdir(category_path) if img.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))])
            category_counts[category] = image_count

    # 类别和对应的数量
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    # 为每个类别生成随机颜色
    colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(len(categories))]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 绘制二维柱状图
    plt.figure(figsize=(10, 10))  # 调整图表大小以适应更多类别
    bars = plt.barh(categories, counts, color=colors)

    # 在每个柱子内右对齐显示具体的数量，使用加粗字体
    for bar in bars:
        width = bar.get_width()
        plt.text(width - 0.05, bar.get_y() + bar.get_height() / 2, f'{int(width)}', ha='right', va='center',
                 color='white', fontweight='bold', fontsize=12)

    # 设置轴标签
    plt.xlabel('图片数量', fontsize=18)
    plt.ylabel('类别名称', fontsize=18)
    # plt.title('数据分布柱状图')

    save_dir = './数据分布柱状图.png'
    plt.savefig(save_dir, dpi=500, bbox_inches='tight')
    # 显示图形
    plt.show()

