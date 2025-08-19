import matplotlib.pyplot as plt
import numpy as np

# 创建图形窗口并设置大小（这里设为12英寸宽、10英寸高）
plt.figure(figsize=(8, 8))

grid = plt.GridSpec(4, 3, wspace=0.6, hspace=0.6, height_ratios=[2, 1, 1, 1], width_ratios=[1, 1, 1])

# 创建3D子图
ax = plt.subplot(grid[0, :], projection='3d')
ax.grid(False)
ax.set_title('3D子图')

# 绘制3D数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
ax.plot_surface(X, Y, Z, cmap='viridis')

plt.show()