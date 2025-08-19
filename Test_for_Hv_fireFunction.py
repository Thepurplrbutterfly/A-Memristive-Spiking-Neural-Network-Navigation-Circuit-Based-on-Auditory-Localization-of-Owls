import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def gaussian_term(x, xi, sigma1=3.88):
    """计算高斯项G(x) = 20*exp(-|x-xi|^2/(2*sigma1^2))"""
    return 20 * np.exp(-np.abs(x - xi)**2 / (2 * sigma1**2))

def correction_term(x, xi):
    """计算修正项F_x(x)，根据不同区间有不同的表达式"""
    dx = np.abs(x - xi)
    result = np.zeros_like(dx)
    
    # 0.5 <= |x-xi| <= 3.5 的情况
    mask1 = (dx >= 0.5) & (dx <= 3.5)
    result[mask1] = - (0.8 - ((dx[mask1] - 0.5) / 5.5))
    
    # 4 <= |x-xi| <= 9.5 的情况
    mask2 = (dx >= 4) & (dx <= 9.5)
    result[mask2] = 2.5 * np.exp(-(dx[mask2] - 7)**2 / 6)
    
    # |x-xi| = 0 或 10 的情况（已包含在初始化的全零数组中）
    
    return result

def f(x, xi):
    """计算函数f(x) = trunc(G(x) + F_x(x))"""
    g = gaussian_term(x, xi)
    f_x = correction_term(x, xi)
    return np.trunc(g + f_x)

def main():
    # 设置参数
    xi = 0  # 中心点
    x_values = np.arange(-10, 10.5, 0.5)  # 区间[-10,10]，步长0.5
    
    # 计算函数值
    g_values = gaussian_term(x_values, xi)
    f_x_values = correction_term(x_values, xi)
    sum_values = g_values + f_x_values
    f_values = f(x_values, xi)
    
    # 打印计算结果
    print("x\tG(x)\tF_x(x)\tG(x)+F_x(x)\tf(x)")
    for x, g, fx, s, f_val in zip(x_values, g_values, f_x_values, sum_values, f_values):
        print(f"{x:.1f}\t{g:.4f}\t{fx:.4f}\t{s:.4f}\t{f_val:.1f}")
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 绘制高斯项 - 折线图
    plt.subplot(2, 2, 1)
    plt.plot(x_values, g_values, 'b-', linewidth=2, marker='o', markersize=5, markevery=2)
    plt.title('高斯项 G(x)', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('G(x)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制修正项 - 折线图
    plt.subplot(2, 2, 2)
    plt.plot(x_values, f_x_values, 'g-', linewidth=2, marker='s', markersize=5, markevery=2)
    plt.title('修正项 F_x(x)', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('F_x(x)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制G(x) + F_x(x) - 折线图
    plt.subplot(2, 2, 3)
    plt.plot(x_values, sum_values, 'm-', linewidth=2, marker='^', markersize=5, markevery=2)
    plt.title('G(x) + F_x(x)', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('G(x) + F_x(x)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制f(x) - 折线图
    plt.subplot(2, 2, 4)
    plt.plot(x_values, f_values, 'r-', linewidth=2, marker='D', markersize=5, markevery=2)
    plt.title('f(x) = trunc(G(x) + F_x(x))', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=3.0)  # 增加子图间距
    plt.savefig('function_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()    