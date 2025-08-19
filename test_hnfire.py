import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from math import pi
from math import sin
from math import cos
from math import asin
from math import atan

def Hn_FIT(M_ITD, Num_CD, HAmp, CD_field, Mod_var):
    #HAmp = 1000
    #Num_CD = 19
    f_max= 20
    sigma_f = CD_field #3.88
    var_mod = Mod_var #6
    HF_rate = np.zeros((1, Num_CD))  
    lr_ITD = M_ITD   #转换为毫秒
    """
    注意:lr_ITD(ms)取值区间为[-180,180],且步长为1
    """
    if np.abs(lr_ITD) > 0.18 * HAmp:  
        print("lr_ITD exceeds the expected range!")
        return HF_rate
    elif np.abs(lr_ITD) <= 0.18 * HAmp: 
        """
        对入耳时间差分三种情况讨论
        """
        if lr_ITD > 0:  #声源位于(0°，90°]区间
            if lr_ITD % 20 == 0:  #此时水平角为10倍数角
                H_Angle_r = (((lr_ITD / 2) / 180) * pi) * (180 / pi)
                H_Angle_opt = ((int(int(lr_ITD / 2) / 10) * 10 / 180) * pi) * (180 / pi)
                HF_rate[0, 9 + int(H_Angle_opt / 10)] = int(f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_opt),2) / (2 * math.pow(sigma_f,2))))
            elif lr_ITD % 20 != 0:   #此时水平角位于两个10倍数角之间
                H_Angle_r =  (((lr_ITD / 2) / 180) * pi) * (180 / pi)
                H_Angle_Ropt = ((np.ceil((lr_ITD / 2) / 10) * 10 / 180) * pi) * (180 / pi)
                H_Angle_Lopt = ((int(int(lr_ITD / 2) / 10) * 10 / 180) * pi) * (180 / pi)
                if abs(H_Angle_r - H_Angle_Ropt) >= 0.5 and abs(H_Angle_r - H_Angle_Ropt) <= 3.5:
                    HF_rate[0, 9 + int(H_Angle_Ropt / 10)] = int((f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_Ropt),2) / (2 * math.pow(sigma_f,2)))) + ((((abs(H_Angle_r - H_Angle_Ropt)) - 0.5) / 5.5) - 0.8))
                if abs(H_Angle_r - H_Angle_Ropt) >=4 and abs(H_Angle_r - H_Angle_Ropt) <= 9.5:
                    HF_rate[0, 9 + int(H_Angle_Ropt / 10)] = int((f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_Ropt),2) / (2 * math.pow(sigma_f,2)))) + (4.4 * math.exp(-math.pow((abs(H_Angle_r - H_Angle_Ropt) - 6),2) / var_mod)))
                if abs(H_Angle_r - H_Angle_Lopt) >= 0.5 and abs(H_Angle_r - H_Angle_Lopt) <= 3.5:
                    HF_rate[0, 9 + int(H_Angle_Lopt / 10)] = int((f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_Lopt),2) / (2 * math.pow(sigma_f,2)))) + ((((abs(H_Angle_r - H_Angle_Lopt)) - 0.5) / 5.5) - 0.8))
                if abs(H_Angle_r - H_Angle_Lopt) >=4 and abs(H_Angle_r - H_Angle_Lopt) <= 9.5:
                    HF_rate[0, 9 + int(H_Angle_Lopt / 10)] = int((f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_Lopt),2) / (2 * math.pow(sigma_f,2)))) + (4.4 * math.exp(-math.pow((abs(H_Angle_r - H_Angle_Lopt) - 6),2) / var_mod)))
        elif lr_ITD < 0:  #声源位于[-90°，0°)区间
            if lr_ITD % 20 == 0:  #此时水平角为10倍数角
                H_Angle_r = (((lr_ITD / 2) / 180) * pi) * (180 / pi)
                H_Angle_opt = ((int(int(lr_ITD / 2) / 10) * 10 / 180) * pi) * (180 / pi)
                HF_rate[0, 9 + int(H_Angle_opt / 10)] = int(f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_opt),2) / 2 * math.pow(sigma_f,2)))
            elif lr_ITD % 20 != 0:   #此时水平角位于两个10倍数角之间
                H_Angle_r =  (((lr_ITD / 2) / 180) * pi) * (180 / pi)
                H_Angle_Lopt = ((np.floor((lr_ITD / 2) / 10) * 10 / 180) * pi) * (180 / pi)
                H_Angle_Ropt = ((int(int(lr_ITD / 2) / 10) * 10 / 180) * pi) * (180 / pi)
                if abs(H_Angle_r - H_Angle_Ropt) >= 0.5 and abs(H_Angle_r - H_Angle_Ropt) <= 3.5:
                    HF_rate[0, 9 + int(H_Angle_Ropt / 10)] = int((f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_Ropt),2) / 2 * math.pow(sigma_f,2))) + ((((abs(H_Angle_r - H_Angle_Ropt)) - 0.5) / 5.5) - 0.8))
                if abs(H_Angle_r - H_Angle_Ropt) >=4 and abs(H_Angle_r - H_Angle_Ropt) <= 9.5:
                    HF_rate[0, 9 + int(H_Angle_Ropt / 10)] = int((f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_Ropt),2) / 2 * math.pow(sigma_f,2))) + (4.4 * math.exp(-math.pow((abs(H_Angle_r - H_Angle_Ropt) - 6),2) / var_mod)))
                if abs(H_Angle_r - H_Angle_Lopt) >= 0.5 and abs(H_Angle_r - H_Angle_Lopt) <= 3.5:
                    HF_rate[0, 9 + int(H_Angle_Lopt / 10)] = int((f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_Lopt),2) / 2 * math.pow(sigma_f,2))) + ((((abs(H_Angle_r - H_Angle_Lopt)) - 0.5) / 5.5) - 0.8))
                if abs(H_Angle_r - H_Angle_Lopt) >=4 and abs(H_Angle_r - H_Angle_Lopt) <= 9.5:
                    HF_rate[0, 9 + int(H_Angle_Lopt / 10)] = int((f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_Lopt),2) / 2 * math.pow(sigma_f,2))) + (4.4 * math.exp(-math.pow((abs(H_Angle_r - H_Angle_Lopt) - 6),2) / var_mod)))
        elif lr_ITD == 0:  #声源位于0°
            H_Angle_r = (((lr_ITD / 2) / 180) * pi) * (180 / pi)
            H_Angle_opt = ((int(int(lr_ITD / 2) / 10) * 10 / 180) * pi) * (180 / pi)
            HF_rate[0, 9] = int(f_max * math.exp(-math.pow(abs(H_Angle_r - H_Angle_opt),2) / 2 * math.pow(sigma_f,2)))
    return HF_rate

def Hn_Fire(T_h, HF_rate):
    """
    模拟神经元在给定时间步长内的固定次数放电情况
    
    参数:
    T_h (int): 总时间
    HF_rate (numpy.ndarray): 1xN的数组，表示N个神经元的总放电次数
    
    返回:
    numpy.ndarray: Nx2xT的数组，记录每个神经元在每个时间步的状态
    """
    time_step = 0.5
    hF_rate_vector = HF_rate
    N_h = hF_rate_vector.shape[1]
    total_steps = int(T_h / time_step)
    hN_Fire = np.zeros((N_h, 2, total_steps))   # 创建一个矩阵记录N个神经元在T_h时间内的放电情况
    
    for n in range(N_h):
        f_counter = 0
        for t in range(total_steps):
            hN_Fire[n, 0, t] = t * time_step  # 记录实际时间
            f_maxN = HF_rate[0, n]
            if f_counter < f_maxN:
                hN_Fire[n, 1, t] = 1   # 二值放电
                f_counter += 1
            else:
                hN_Fire[n, 1, t] = 0.1   # 非放电神经元输出
    
    return hN_Fire

def visualize_neuron_firing(results, hf_rate, title="神经元放电可视化"):
    """使用线条可视化神经元放电结果，突出显示特定神经元"""
    N_h, _, total_steps = results.shape
    
    # 计算每个神经元的实际放电次数
    fire_counts = np.sum(results[:, 1, :] == 1, axis=1)
    
    # 创建一个更大的图形
    plt.figure(figsize=(16, 10), dpi=150)
    
    # 设置中文字体支持
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    
    # 绘制每个神经元的放电情况
    for n in range(N_h):
        # 获取该神经元的放电时间点
        fire_times = np.where(results[n, 1, :] == 1)[0]
        
        # 计算设定的放电次数
        set_fire_count = hf_rate[0, n]
        
        # 如果有放电，绘制垂直线
        if len(fire_times) > 0:
            # 对有放电次数的神经元使用不同颜色
            if set_fire_count > 0:
                color = plt.cm.tab20(n % 20)  # 使用不同颜色区分神经元
            else:
                color = 'gray'  # 无放电次数的神经元用灰色
            
            # 为每个放电时间绘制一条垂直线
            for t in fire_times:
                plt.plot([t], [n], 'o', color=color, markersize=5)
        
        # 在右侧标注设定放电次数和实际放电次数
        plt.text(total_steps + 2, n, f'{set_fire_count} ({int(fire_counts[n])})', 
                 ha='left', va='center', fontsize=9, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # 高亮显示第7和第8个神经元
    highlight_neurons = [6, 7]  # 索引为6和7的神经元（第7和第8个）
    for n in highlight_neurons:
        plt.axhspan(n-0.5, n+0.5, color='yellow', alpha=0.3)
    
    # 设置坐标轴
    plt.yticks(range(N_h))
    plt.xlim(-5, total_steps + 15)  # 增加右侧空间用于标注
    plt.ylim(-0.5, N_h - 0.5)
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel('时间步 (每个间隔=0.5)', fontsize=14)
    plt.ylabel('神经元索引', fontsize=14)
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='-', alpha=0.3)
    
    # 创建自定义图例
    custom_lines = [
        Line2D([0], [0], color='blue', marker='o', linestyle='', markersize=5),
        Line2D([0], [0], color='gray', marker='o', linestyle='', markersize=5),
        Line2D([0], [0], color='yellow', lw=10, alpha=0.3)
    ]
    
    plt.legend(custom_lines, ['放电', '无放电次数', '重点观察神经元'], loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    plt.show()

def main():
    # 设置参数
    T_h = 10  # 总时间
    M_ITD = 86
    Num_CD = 19
    HAmp = 1000
    CD_field = 3.88
    Mod_var = 6
    # 定义1行19列的放电次数数组
    HF_rate = Hn_FIT(M_ITD, Num_CD, HAmp, CD_field, Mod_var)
    print(f"HF_rate:{HF_rate}")
    # 运行模拟
    results = Hn_Fire(T_h, HF_rate)
    Ex = results[:, 1, :]
    print(f"HF_fire:{Ex}")
    # 打印统计信息
    print("=== 神经元放电统计 ===")
    for n in range(19):  # 打印所有神经元
        fire_count = np.sum(results[n, 1, :] == 1)
        print(f"神经元 {n+1}: 设定放电次数 {HF_rate[0, n]}, 实际放电次数 {int(fire_count)}")
    
    # 可视化结果
    visualize_neuron_firing(results, HF_rate, "神经元固定次数放电模式")

if __name__ == "__main__":
    main()