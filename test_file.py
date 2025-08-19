import numpy as np

theta_degrees = np.arange(-90, 90.5, 0.5)  # 生成-90到90度，步长0.5度的角度数组
delta_T = np.sin(np.radians(theta_degrees)) / 1.7  # 计算对应的ΔT值（单位：ms）

# 保存数据到CSV文件
np.savetxt('delta_T_values.csv', np.column_stack((theta_degrees, delta_T)), 
           delimiter=',', header='theta_degrees,delta_T_ms', comments='')

# 打印前10个和后10个值
print("θ (度)\tΔT (ms)")
print("------------------")
for i in range(10):
    print(f"{theta_degrees[i]:.1f}\t{delta_T[i]:.4f}")
print("...")
for i in range(-10, 0):
    print(f"{theta_degrees[i]:.1f}\t{delta_T[i]:.4f}")