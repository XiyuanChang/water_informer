import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 数据的类别
categories = ['A', 'B', 'C', 'D', 'E']

# 多组数据
values_list = [
    [4, 3, 2, 5, 4],  # 第一组
    [3, 2, 4, 3, 5],  # 第二组
    [5, 4, 3, 2, 1]   # 第三组
]

# 计算角度
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # 闭合角度

# 创建图形，设置 A 在正上方，起始角度90度
fig = plt.figure(figsize=(8, 8))  # 创建一个图形
ax = plt.subplot(111, polar=True)  # 添加极坐标子图
ax.set_theta_offset(pi / 2)  # 设置角度偏移量为 90 度
ax.set_theta_direction(-1)   # 逆时针方向

# 绘制背景的多个虚线五边形
for i in range(1, 6):  # 画出五个虚线五边形
    # 计算背景虚线五边形的坐标
    values_background = [i] * N
    values_background += values_background[:1]  # 闭合数据
    ax.plot(angles, values_background, linestyle='--', color='gray', alpha=0.3)

# 绘制从中心到 A、B、C、D、E 的线条，增加线宽
for angle in angles[:-1]:
    ax.plot([angle, angle], [0, 5], color='gray', linewidth=2)  # 从中心到每个点的线条，线宽设为2

# 绘制每组数据
colors = ['b', 'g', 'r']  # 为不同组设置颜色
labels = ['Group 1', 'Group 2', 'Group 3']  # 图例标签

for i, values in enumerate(values_list):
    values += values[:1]  # 闭合数据
    ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=labels[i])  # 用实线绘制数据
    ax.fill(angles, values, color=colors[i], alpha=0.1)  # 填充五边形区域

# 设置类别标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# 去除外部圆形边框和圆形网格
ax.spines['polar'].set_visible(False)  # 去掉外边的圆形边框
ax.yaxis.set_visible(False)  # 隐藏圆形的径向坐标轴
ax.xaxis.grid(True)  # 保留从中心辐射出的坐标轴线

# 添加图例
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
# 保存图形为图片，格式为PNG
plt.savefig('radar_chart_with_multiple_groups_corrected.png', dpi=300, bbox_inches='tight')  # 保存为文件
plt.show()
