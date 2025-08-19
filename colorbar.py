import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
# 生成示例数据
np.random.seed(42)
W_v = np.random.rand(19, 100) * 19  # 0-100 的随机数

# 自定义颜色映射（蓝-白-红）
cmap = LinearSegmentedColormap.from_list('_3_colors', ['mediumspringgreen', 'lavender','b'])
cmap2 = LinearSegmentedColormap.from_list('_6_colors', ['springgreen', 'mediumseagreen', 'c', 'dodgerblue', 'royalblue' ,'b'])
cmap3 = LinearSegmentedColormap.from_list('_5_colors', ['palevioletred', 'thistle', 'azure', '#B3B3FF', 'navy'])
cmap4 = LinearSegmentedColormap.from_list('_5_colors', ['#F4F400', '#FDE720', '#FCCB21', '#F7B826', '#EB9C0A', '#EC7509', '#E4510A', '#C21467', '#C1159F', '#A9128B', '#9413A2', '#7D118C', '#711399', '#6718A0', '#4B18A0'])
cmap5 = LinearSegmentedColormap.from_list('_5_colors', ['ghostwhite', 'lavender', 'lightskyblue', 'skyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'blue', 'mediumblue', 'navy'])
cmap6 = LinearSegmentedColormap.from_list('_5_colors', ['mintcream', 'honeydew', 'palegreen', 'lightgreen', 'lawngreen', 'springgreen', 'lime', 'limegreen', 'mediumseagreen', 'green'])
cmap7 = LinearSegmentedColormap.from_list('_5_colors', ['lavender', 'dodgerblue', 'blue'])
cmap8 = LinearSegmentedColormap.from_list('_5_colors', ['papayawhip', 'gold', 'orange'])
cmap9 = LinearSegmentedColormap.from_list('_5_colors', ['lavender', 'blue'])
cmap10 = LinearSegmentedColormap.from_list('_5_colors', ['papayawhip', 'orange'])
cmap11 = LinearSegmentedColormap.from_list('CD_neurons_ini_W_colors', ['#FEF9CB', '#FDF3A2', '#FCEE74', '#FBE731','#FBD831', '#FBD006',  'orange', 'darkorange', '#F16D1F', '#ED5A14'])
# 绘制热力图
plt.rcParams["font.family"] = "Times New Roman"
fig=plt.figure(figsize=(10, 12))
ax_Hf = sns.heatmap(W_v, annot=False, cmap=cmap11, xticklabels=False, yticklabels=False,
                                linewidths=0,  square=True, cbar_kws={'label': 'Fire Intensity', 'orientation': 'horizontal', 'pad': 0.1})  # 颜色条标签
# 获取颜色条对象并设置标签
cbar = ax_Hf.collections[0].colorbar
cbar.set_label('Firing Intensity', fontsize=22)  # 水平显示标签
cbar.ax.tick_params(labelsize=22)  # 颜色条刻度字体大小
# 调整颜色条位置和大小
cbar.ax.set_position([0.15, 0.2, 0.725, 0.03])  # [左, 下, 宽, 高]
# 设置x轴刻度位置和标签（区间从1到100）
x_positions = np.arange(0.5, 100, 2)  # 刻度位置（每个刻度在网格中间）
x_labels = np.arange(1, 100 + 1, 2)        # 刻度标签（从1到100，步长10）
ax_Hf.set_xticks(x_positions)
ax_Hf.set_xticklabels(x_labels)
y_positions = np.arange(0.5, 19, 2)  # 刻度位置（每个刻度在网格中间）
y_labels = np.arange(1, 20, 2)        # 刻度标签（从1到100，步长10）
ax_Hf.set_yticks(y_positions)
ax_Hf.set_yticklabels(y_labels)
plt.xlabel('Time(ms)', fontsize=22)  # x轴标签字体大小
plt.xticks(fontsize=15)  # x轴刻度字体大小
plt.ylabel('CD Neuron', fontsize=22)  # y轴标签字体大小
plt.yticks(fontsize=15)  # y轴刻度字体大小
plt.title('CD Neurons Firing Intensity', fontsize=22)
plt.show()