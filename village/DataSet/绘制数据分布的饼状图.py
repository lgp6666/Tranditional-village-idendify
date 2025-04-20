import os
import matplotlib.pyplot as plt
import numpy as np

# 文件夹路径
folder_path = 'ori'

# 获取每个类别的图片数量
category_counts = {}
for category in os.listdir(folder_path):
    category_path = os.path.join(folder_path, category)
    if os.path.isdir(category_path):
        image_count = len([img for img in os.listdir(category_path) if img.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))])
        category_counts[category] = image_count

# 类别和对应的数量
categories = list(category_counts.keys())
counts = list(category_counts.values())

# 计算每个类别的百分比
total_counts = sum(counts)
percentages = [(count / total_counts) * 100 for count in counts]

# 生成随机颜色
colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(len(categories))]
plt.rcParams['font.sans-serif'] = ['SimHei']
# 绘制二维饼状图
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(percentages, colors=colors, startangle=140, autopct='%1.1f%%')

# 更新图注内容，包含类别和百分比
legend_labels = [f'{category}: {percentage:.2f}%' for category, percentage in zip(categories, percentages)]
ax.legend(wedges, legend_labels, title="类别", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# plt.title('数据分布饼状图', fontsize=32, fontweight='bold')

save_dir = './数据分布饼状图.png'
plt.savefig(save_dir, dpi=500, bbox_inches='tight')
# 显示图形
plt.show()
