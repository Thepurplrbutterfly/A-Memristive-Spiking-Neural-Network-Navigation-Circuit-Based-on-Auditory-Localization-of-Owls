import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from math import pi
from math import sin
from math import cos
from math import asin
from math import atan

def Vn_FIT(M_ILD, Num_IC, VAmp):
    #VAmp = 10
    #Num_IC = 19
    VF_rate = np.zeros((1, Num_IC))  #19个垂直角检测神经元构成1×19矩阵，每个元素表示对应神经元放电强度
    lr_ILD = M_ILD  #电压表示入耳强度差:[0.1, 0.9] mapping to [10°, 90°] [-0.1, -0.9] mapping to [-10°, -90°]
    """
    注意:lr_ILD(V)取值区间为[-0.9,0.9],且步长为0.01
    """
    #fmax = 100   #最大放电数
    #theta_r = 0    #实际声源垂直角
    #theta_opt = 0    #10倍数垂直角
    #sigma = 1
    if lr_ILD > 0.9 and lr_ILD < -0.9:
        print("lr_ILD exceeds the expected range!")
        return VF_rate
    elif lr_ILD <= 0.9 and lr_ILD >= -0.9:
        """
        对声强差分三种情况进行讨论
        """
        if lr_ILD > 0:   #声源位于(0°，90°]区间
            if ((lr_ILD * VAmp) % 1) == 0:   #此时垂直角为10倍数角
                V_Angle_r = (((lr_ILD * VAmp * 10) / 180) * pi) * (180 / pi)
                V_Angle_opt = ((int(lr_ILD * VAmp) * 10) / 180) * pi * (180 / pi) 
                VF_rate[0, 9 + int(lr_ILD * VAmp)] = 1 * (V_Angle_r / V_Angle_opt)
            elif ((lr_ILD * VAmp) % 1) != 0:    #此时垂直角位于两个10倍数角之间
                V_theta_r = (((lr_ILD * VAmp * 10) / 180) * pi) * (180 / pi)
                V_theta_Lopt = (((int(lr_ILD * VAmp) * 10) /180) * pi) * (180 / pi)  #比theta_r小的邻近10倍数垂直角
                V_theta_Ropt = (((math.ceil(lr_ILD * VAmp) * 10) /180) * pi) * (180 / pi)   #比theta_r大的邻近10倍数垂直角
                VF_rate[0, 9 + int(lr_ILD * VAmp)] = 1 * (1 - ((abs(V_theta_r - V_theta_Lopt)) / 10))
                VF_rate[0, 9 + math.ceil(lr_ILD * VAmp)] = 1 * (1 - ((abs(V_theta_r - V_theta_Ropt)) / 10))
        elif lr_ILD < 0:   #声源位于[-90°，0°)区间
            if ((np.abs(lr_ILD) * VAmp) % 1) == 0: 
                V_Angle_r = (((lr_ILD * VAmp * 10) / 180) * pi) * (180 / pi)
                V_Angle_opt = ((int(lr_ILD * VAmp) * 10) / 180) * pi * (180 / pi)
                VF_rate[0, 9 - int(np.abs(lr_ILD) * VAmp)] = 1 * (V_Angle_r / V_Angle_opt)
            elif ((np.abs(lr_ILD) * VAmp) % 1) != 0:    #此时垂直角位于两个10倍数角之间
                V_theta_r = (((lr_ILD * VAmp * 10) / 180) * pi) * (180 / pi)
                V_theta_Ropt = (((int(lr_ILD * VAmp) * 10) /180) * pi) * (180 / pi)   #比theta_r大的邻近10倍数垂直角
                V_theta_Lopt = (((math.floor(lr_ILD * VAmp) * 10) /180) * pi) * (180 / pi)   #比theta_r小的邻近10倍数垂直角
                VF_rate[0, 9 - int(np.abs(lr_ILD) * VAmp)] != 1 * (1 - ((abs(V_theta_r - V_theta_Ropt)) / 10))
                VF_rate[0, 9 - math.ceil(np.abs(lr_ILD) * VAmp)] = 1 * (1 - ((abs(V_theta_r - V_theta_Lopt)) / 10))
        elif lr_ILD == 0:   #声源位于0°
            V_Angle_r = (((lr_ILD * VAmp * 10) / 180) * pi) * (180 / pi)
            V_Angle_opt = ((int(lr_ILD * VAmp) * 10) / 180) * pi * (180 / pi) 
            VF_rate[0, 9 - int(lr_ILD * VAmp)] = 1 * (V_Angle_r / V_Angle_opt)
            #VF_rate[0, 9] = 1 * (V_Angle_r / V_Angle_opt)
    return VF_rate

def Vn_Fire(T_v, VF_rate):
    """
    模拟神经元在给定时间步长内的放电情况
    
    参数:
    T_v (int): 时间步长数
    VF_rate (numpy.ndarray): 1xN的数组，表示N个神经元的放电率
    
    返回:
    numpy.ndarray: Nx2xT_v的数组，记录每个神经元在每个时间步的状态
    """
    vF_rate_vector = VF_rate
    N_v = vF_rate_vector.shape[1]
    vN_Fire = np.zeros((N_v, 2, T_v))   # 创建一个矩阵记录N个神经元在T_v步长内的放电情况
    
    for t in range(T_v):
        for n in range(N_v):
            vN_Fire[n, 0, t] = t  # 记录时间步
            if vF_rate_vector[0, n] >= np.random.random(): 
                vN_Fire[n, 1, t] = 1  # 二值放电
            else:
                vN_Fire[n, 1, t] = 0.1  # 非放电神经元输出
    
    return vN_Fire

def visualize_neuron_firing(results, vf_rate, title="神经元放电可视化"):
    """使用线条可视化神经元放电结果，突出显示特定神经元"""
    N_v, _, T_v = results.shape
    
    # 计算每个神经元的放电次数
    fire_counts = np.sum(results[:, 1, :] == 1, axis=1)
    
    # 创建一个更大的图形
    plt.figure(figsize=(16, 10), dpi=150)
    
    # 设置中文字体支持
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    # 绘制每个神经元的放电情况
    for n in range(N_v):
        # 获取该神经元的放电时间点
        fire_times = np.where(results[n, 1, :] == 1)[0]
        
        # 计算放电率
        rate = vf_rate[0, n]
        
        # 如果有放电，绘制垂直线
        if len(fire_times) > 0:
            # 对有放电率的神经元使用不同颜色
            if rate > 0:
                color = plt.cm.tab20(n % 20)  # 使用不同颜色区分神经元
            else:
                color = 'gray'  # 无放电率的神经元用灰色
            
            # 为每个放电时间绘制一条垂直线
            for t in fire_times:
                plt.plot([t, t], [n-0.3, n+0.3], color=color, linewidth=1.5)
        
        # 在右侧标注放电次数
        plt.text(T_v + 2, n, f'{rate:.1f} ({fire_counts[n]})', 
                 ha='left', va='center', fontsize=9, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # 高亮显示第7和第8个神经元
    highlight_neurons = [6, 7]  # 索引为6和7的神经元（第7和第8个）
    for n in highlight_neurons:
        plt.axhspan(n-0.5, n+0.5, color='yellow', alpha=0.3)
    
    # 设置坐标轴
    plt.yticks(range(N_v))
    plt.xlim(-5, T_v + 15)  # 增加右侧空间用于标注
    plt.ylim(-0.5, N_v - 0.5)
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel('时间步', fontsize=14)
    plt.ylabel('神经元索引', fontsize=14)
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # 创建自定义图例
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='gray', lw=2),
        Line2D([0], [0], color='yellow', lw=10, alpha=0.3)
    ]
    
    plt.legend(custom_lines, ['放电', '无放电率', '重点观察神经元'], loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    plt.show()

def main():
    # 设置参数
    T_v = 100  # 时间步长
    M_ILD = 0.23
    Num_IC = 19
    VAmp = 10
    # 定义1行19列的放电率数组
    VF_rate = Vn_FIT(M_ILD, Num_IC, VAmp)
    print(f"HF_rate:{VF_rate}")
    # 运行模拟
    results = Vn_Fire(T_v, VF_rate)
    Ex = results[:, 1, :]
    print(f"VF_Fire:{Ex}")
    
    # 打印统计信息
    print("=== 神经元放电统计 ===")
    for n in range(19):  # 只打印第7和第8个神经元
        fire_ratio = np.mean(results[n, 1, :] == 1)
        fire_count = np.sum(results[n, 1, :] == 1)
        print(f"神经元 {n+1}: 设定放电率 {VF_rate[0, n]:.1f}, 实际放电率 {fire_ratio:.2%}, 放电次数 {fire_count}")
    
    # 可视化结果
    visualize_neuron_firing(results, VF_rate, "神经元放电模式 (垂直线表示放电时间点)")

if __name__ == "__main__":
    main()