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
                hN_Fire[n, 1, t] = 1e-20   # 非放电神经元输出
    
    return hN_Fire

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
                vN_Fire[n, 1, t] = 1e-20  # 非放电神经元输出
    
    return vN_Fire

W_h = np.random.uniform(0,1,(19,19))
W_v = np.random.uniform(0,1,(19,19))
W_h = (((W_h - np.min(W_h)) / (np.max(W_h) - np.min(W_h))) * 15 - 8).round()
W_v = (((W_v - np.min(W_v)) / (np.max(W_v) - np.min(W_v))) * 15 - 8).round()
W_h = W_h + 8
W_v = W_v + 8

def population_encoding(H_multiangles, V_multiangles):
    #H_multiangles:水平角输出神经元输出矩阵(1×19)，其中每个元素为1个T_h步长内水平角检测神经元输入与权重的乘积和
    #V_multiangles:垂直角输出神经元输出矩阵(1×19)，其中每个元素为1个T_v步长内垂直角检测神经元输入与权重的乘积和
    #H_multiangles和V_multiangles均为忆阻计算平台输出
    encoded_HAngle = 0   #群体编码后的水平角
    encoded_VAngle = 0   #群体编码后的垂直角
    SIN_H = 0
    COS_H = 0
    SIN_V = 0
    COS_V = 0
    N_h = H_multiangles.shape[1]
    N_v = V_multiangles.shape[1]
    Nml_H = (H_multiangles - np.min(H_multiangles)) / (np.max(H_multiangles) - np.min(H_multiangles))
    Nml_V = (V_multiangles - np.min(V_multiangles)) / (np.max(V_multiangles) - np.min(V_multiangles))
    for i in range(N_h):
        SIN_H = sin(((i - 9) / 18) * pi) * Nml_H[0, i] + SIN_H
        COS_H = cos(((i - 9) / 18) * pi) * Nml_H[0, i] + COS_H
    encoded_HAngle = atan(SIN_H / COS_H)
    for i in range(N_v):
        SIN_V = sin(((i - 9) / 18) * pi) * Nml_V[0, i] + SIN_V
        COS_V = cos(((i - 9) / 18) * pi) * Nml_V[0, i] + COS_V
    encoded_VAngle = atan(SIN_V / COS_V)
    ec_Angle_Array = np.round(np.array([encoded_HAngle, encoded_VAngle], dtype=np.float32), 2)  #用于存储估计的水平角encoded_HAngle和估计的垂直角encoded_VAngle的数组
    return ec_Angle_Array 

"""
def WTA_flag_update(H_multiangles, V_multiangles):
    #将两个输出神经元矩阵中与最大值元素相同的元素置1，以此作为赢家标志，并返回对应的二值矩阵
    #print(f"H_multiangles 形状: {H_multiangles.shape}") 
    H_flag = (H_multiangles == np.max(H_multiangles)).astype(int)
    V_flag = (V_multiangles == np.max(V_multiangles)).astype(int)
    #将两个标志矩阵统一存储
    Flag_Array = np.array([H_flag, V_flag])
    return Flag_Array
"""
def WTA_flag_update(H_multiangles, V_multiangles):
    #将两个输出神经元矩阵中大小排前三的元素置1，以此作为赢家标志，并返回对应的二值矩阵
    H_SQ = np.unique(H_multiangles)[::-1]   #将H_multiangles中的输出进行去重并从小到大排序，[::-1]进行倒序，最终得到从大到小的排序
    H_top3 = H_SQ[:3]   #取H_SQ中前三个最大数
    H_flag = np.isin(H_multiangles, H_top3).astype(int)   #将H_multiangles中对应前三个最大元素的位置置1
    V_SQ = np.unique(V_multiangles)[::-1]   #将H_multiangles中的输出进行去重并从小到大排序，[::-1]进行倒序，最终得到从大到小的排序
    V_top3 = V_SQ[:3]   #取H_SQ中前三个最大数
    V_flag = np.isin(V_multiangles, V_top3).astype(int)   #将H_multiangles中对应前三个最大元素的位置置1
    #将两个标志矩阵统一存储
    Flag_Array = np.array([H_flag, V_flag])
    return Flag_Array

def main():
    # 设置参数
    T_h = 10  # 总时间
    M_ITD = 86
    Num_CD = 19
    HAmp = 1000
    CD_field = 3.88
    Mod_var = 6
    
    T_v = 100  # 时间步长
    M_ILD = 0.23
    Num_IC = 19
    VAmp = 10
    dt_step = 1
    HF_rate = Hn_FIT(M_ITD, Num_CD, HAmp, CD_field, Mod_var)
    print(f"HF_rate:{HF_rate}")
    # 运行模拟
    X_h = Hn_Fire(T_h, HF_rate)
    
    VF_rate = Vn_FIT(M_ILD, Num_IC, VAmp)
    print(f"HF_rate:{VF_rate}")
    # 运行模拟
    X_v = Vn_Fire(T_v, VF_rate)
    
    t_h = np.arange(1, T_h*2+1, dt_step)
    t_v = np.arange(1, T_v+1, dt_step)
    HCD_CPU_Output = np.zeros((Num_CD, len(t_h)))
    VIC_CPU_Output = np.zeros((Num_IC, len(t_v)))
    
    print(f"W_h:{W_h}")
    np.savetxt("W_h_full.txt", W_h, delimiter=',', fmt="%d")
    print(f"W_h.T:{W_h.T}")
    np.savetxt("W_h.T_full.txt", W_h.T, delimiter=',', fmt="%d")
    print(f"W_v:{W_v}")
    np.savetxt("W_v_full.txt", W_v, delimiter=',', fmt="%d")
    print(f"W_v.T:{W_v.T}")
    np.savetxt("W_v.T_full.txt", W_v.T, delimiter=',', fmt="%d")
    print(f"X_h:{X_h[:, 1, :]}")
    np.savetxt("X_h_full.txt", X_h[:, 1, :], delimiter=',', fmt="%d")
    print(f"X_v:{X_v[:, 1, :]}")
    np.savetxt("X_v_full.txt", X_v[:, 1, :], delimiter=',', fmt="%d")
    t_ih = 1
    for t_iv in t_v:
        """由于CD神经元输出计算单步长时间仅为IC神经元输出计算单步长时间的1/10, 所以这里在IC神经元输出计算的前10次遍历的同时对IC神经元输出进行计算"""
        if t_ih <= np.max(t_h):
            H_input_array = X_h[:, 1, t_ih-1].reshape(-1, 1)  #将19个水平角神经元在一个时间步t_ih-1时刻的放电情况存储为一个(19, 1)大小的矩阵
            #==============计算水平角神经元在一个单时间步的加权和=============#
            HCD_CPU = np.dot(W_h.T, H_input_array)
            print(f"HCD_CPU:{HCD_CPU}") 
            #print(f"HCD_CPU 形状: {HCD_CPU.shape}") 
            #t_ih += 1
            HCD_CPU_Output[:, t_ih-1] = np.squeeze(HCD_CPU)
            #print(f"HCD_CPU_Output形状: {HCD_CPU_Output[:, t_ih-1].shape}") 
            t_ih += 1
        V_input_array = X_v[:, 1, t_iv-1].reshape(-1, 1)  #将19个垂直角神经元在一个时间步t_ih-1时刻的放电情况存储为一个(19, 1)大小的矩阵
        #==============计算垂直角神经元在一个单时间步的加权和=============#
        VIC_CPU = np.dot(W_v.T, V_input_array)
        VIC_CPU_Output[:, t_iv-1] = np.squeeze(VIC_CPU)   
    print(f"HCD_CPU_Output:{HCD_CPU_Output}")
    #print(f"HCD_CPU_Output_20:{HCD_CPU_Output[:, 19]}")
    print(f"VIC_CPU_Output:{VIC_CPU_Output}")
    #print(f"VIC_CPU_Output_100:{VIC_CPU_Output[:, 99]}")
    
    HCD_CPU_Output_sum = np.zeros((Num_CD, 1))
    VIC_CPU_Output_sum = np.zeros((Num_IC, 1))
    f_sum_Hn = np.zeros((Num_CD, 1))
    f_sum_Vn = np.zeros((Num_IC, 1))
    #X_h_temp = X_h
    #print(f"f_sum_Hn 形状: {f_sum_Hn.shape}") 
    #X_v_temp = X_v
    tt_ih = 1
    #HCD_MVM_Output_sum_rec = np.zeros((T_max/T_v, Num_CD))
    #VIC_MVM_Output_sum_rec = np.zeros((T_max/T_v, Num_IC))
    """对一个步长内每个单步长的计算结果极性累加"""
    for tt_iv in t_v:
        #==============对水平角神经元在一个时间步内的输入和输出进行累加=============#
        if tt_ih <= np.max(t_h):
            f_sum_Hn = X_h[:, 1, tt_ih-1].reshape(-1, 1) + f_sum_Hn
            #print(f"f_sum_Hn 形状: {f_sum_Hn.shape}") 
            HCD_CPU_Output_sum = HCD_CPU_Output[:, tt_ih-1].reshape(-1, 1) + HCD_CPU_Output_sum
            #print(f"HCD_CPU_Output_sum 形状: {HCD_CPU_Output_sum.shape}") 
            tt_ih += 1
        #==============对垂直角角神经元在一个时间步内的输入和输出进行累加=============#
        f_sum_Vn = X_v[:, 1, tt_iv-1].reshape(-1, 1) + f_sum_Vn
        VIC_CPU_Output_sum = VIC_CPU_Output[:, tt_iv-1].reshape(-1, 1) + VIC_CPU_Output_sum
    HCD_CPU_Output_sum = HCD_CPU_Output_sum.reshape(1, -1)
    HCD_CPU_Output_sum_nor = (HCD_CPU_Output_sum - np.min(HCD_CPU_Output_sum)) / (np.max(HCD_CPU_Output_sum) - np.min(HCD_CPU_Output_sum))
    VIC_CPU_Output_sum = VIC_CPU_Output_sum.reshape(1, -1)
    VIC_CPU_Output_sum_nor = (VIC_CPU_Output_sum - np.min(VIC_CPU_Output_sum)) / (np.max(VIC_CPU_Output_sum) - np.min(VIC_CPU_Output_sum))
    f_sum_Hn = f_sum_Hn.reshape(1, -1)
    f_sum_Vn = f_sum_Vn.reshape(1, -1)
    print(f"HCD_CPU_Output_sum:{HCD_CPU_Output_sum}")
    print(f"HCD_CPU_Output_sum_nor:{HCD_CPU_Output_sum_nor}")
    print(f"VIC_CPU_Output_sum:{VIC_CPU_Output_sum}")
    print(f"VIC_CPU_Output_sum_nor:{VIC_CPU_Output_sum_nor}")
    print(f"f_sum_Hn:{f_sum_Hn}")
    print(f"f_sum_Vn:{f_sum_Vn}")
    encoded_Angles = population_encoding(HCD_CPU_Output_sum, VIC_CPU_Output_sum)
    encoded_HAngle = encoded_Angles[0]
    encoded_VAngle = encoded_Angles[1]
    encoded_HAngle_d = np.degrees(encoded_HAngle)
    encoded_VAngle_d = np.degrees(encoded_VAngle)
    print(f"编码水平角：{encoded_HAngle_d}")
    print(f"编码垂直角：{encoded_VAngle_d}")
    
    Flag_Array = WTA_flag_update(HCD_CPU_Output_sum, VIC_CPU_Output_sum)
    print(f" H_winner_Flag_Array:{Flag_Array[0, :]}")
    print(f" V_winner_Flag_Array:{Flag_Array[1, :]}")
    
if __name__ == "__main__":
    main()