# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import random
import time
import math
import array
from math import sin
from math import cos
from math import asin
from math import pi
from math import atan #单参数形式，参数为常数
#from math import atan2    #双参数形式，参数为象限坐标 如atan2(1,-1)

#垂直角Vertical Angle检测神经元放电强度计算函数
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
                VF_rate[0, 9 + int(lr_ILD * VAmp)] = 1
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
            VF_rate[0, 9 - int(lr_ILD * VAmp)] = 1 
            #VF_rate[0, 9] = 1 * (V_Angle_r / V_Angle_opt)
    return VF_rate

#垂直角Vertical Angle检测神经元放电函数
def Vn_Fire(T_v, VF_rate):
    #Num_IC = 19
    #T_v = 100
    vF_rate_vector = VF_rate
    N_v = vF_rate_vector.shape[1]
    vN_Fire = np.zeros((N_v, 2, T_v))   #创建一个矩阵记录19个神经元在1个T_v步长内的放电情况
    for t in range(T_v):
        for n in range(N_v):
            vN_Fire[n, 0, t] = t
            if vF_rate_vector[0, n] >= random(): 
               vN_Fire[n, 1, t] = 1  #二值放电
            else:
               vN_Fire[n, 1, t] = 1e-20   
    return vN_Fire

#水平角Horizontal Angle检测神经元放电强度计算函数
def Hn_FIT(M_ITD, Num_CD, HAmp, CD_field, Mod_var):
    #HAmp = 1000
    #Num_CD = 19
    f_max = 20
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

#水平角Horizontal Angle检测神经元放电函数
def Hn_Fire(T_h, HF_rate):
   #T_h = 10 
   time_step = 0.5
   hF_rate_vector = HF_rate
   N_h = hF_rate_vector.shape[1]
   hN_Fire = np.zeros((N_h, 2, int(T_h / time_step)))   #创建一个矩阵记录19个神经元在1个T_h步长内的放电情况
   for n in range(N_h):
       f_counter = 0
       for t in range(int(T_h / time_step)):
           hN_Fire[n, 0, t] = t
           f_maxN = HF_rate[0, n]
           if f_counter < f_maxN:
               hN_Fire[n, 1, t] = 1   #二值放电
               f_counter += 1
           else:
               hN_Fire[n, 1, t] = 1e-20   
   return hN_Fire
"""
方法2:采用概率放电
for n in range(N_h):
    f_maxN = HF_rate[:, n]
    for t in range(T_h / time_step):
        hN_Fire[n, 0, t] = t
        if f_maxN >= random.randint(0, 20):
            hN_Fire[n, 1, t] = 1
        else:
            hN_Fire[n, 1, t] = 0
"""
#群体编码函数(垂直角和水平角)
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
    for i in range(N_h):
        SIN_H = sin(((i - 9) / 18) * pi) * H_multiangles[0, i] + SIN_H
        COS_H = cos(((i - 9) / 18) * pi) * H_multiangles[0, i] + COS_H
    encoded_HAngle = atan(SIN_H / COS_H)
    for i in range(N_v):
        SIN_V = sin(((i - 9) / 18) * pi) * V_multiangles[0, i] + SIN_V
        COS_V = cos(((i - 9) / 18) * pi) * V_multiangles[0, i] + COS_V
    encoded_VAngle = atan(SIN_V / COS_V)
    ec_Angle_Array = np.array([encoded_HAngle, encoded_VAngle], dtype=np.float32)   #用于存储估计的水平角encoded_HAngle和估计的垂直角encoded_VAngle的数组
    return ec_Angle_Array 

"""    
#胜者通吃赢家标志更新函数(仅将最大输出神经元的标志置1)
def WTA_flag_update(H_multiangles, V_multiangles):
    #将两个输出神经元矩阵中与最大值元素相同的元素置1,以此作为赢家标志,并返回对应的二值矩阵
    #print(f"H_multiangles 形状: {H_multiangles.shape}") 
    H_flag = (H_multiangles == np.max(H_multiangles)).astype(int)
    V_flag = (V_multiangles == np.max(V_multiangles)).astype(int)
    #将两个标志矩阵统一存储
    Flag_Array = np.array([H_flag, V_flag])
    return Flag_Array
"""

#胜者通吃赢家标志更新函数(将前三输出大小的神经元标志置1)
def WTA_flag_update(H_multiangles, V_multiangles):
    #将两个输出神经元矩阵中大小排前三的元素置1，以此作为赢家标志，并返回对应的二值矩阵
    H_Sorting = np.unique(H_multiangles)[::-1]   #将H_multiangles中的输出进行去重并从小到大排序，[::-1]进行倒序，最终得到从大到小的排序
    H_top3 = H_Sorting[:3]   #取H_SQ中前三个最大数
    H_flag = np.isin(H_multiangles, H_top3).astype(int)   #将H_multiangles中对应前三个最大元素的位置置1
    V_Sorting = np.unique(V_multiangles)[::-1]   #将H_multiangles中的输出进行去重并从小到大排序，[::-1]进行倒序，最终得到从大到小的排序
    V_top3 = V_Sorting[:3]   #取H_SQ中前三个最大数
    V_flag = np.isin(V_multiangles, V_top3).astype(int)   #将H_multiangles中对应前三个最大元素的位置置1
    #将两个标志矩阵统一存储
    Flag_Array = np.array([H_flag, V_flag])
    return Flag_Array

#Agent与声源间的理论水平角和垂直角计算函数
def true_angle_cal(Agent_pos, source_pos):
    Agent_x = Agent_pos[0]
    Agent_y = Agent_pos[1]
    Agent_z = Agent_pos[2]
    source_x = source_pos[0]
    source_y = source_pos[1]
    source_z = source_pos[2]
    #理论垂直角计算(弧度值表示)
    if (Agent_x - source_x == 0) and (Agent_y - source_y == 0):  #case1:垂直角=90°或-90°
        if source_z - Agent_z > 0:
            T_VAngle = pi/2
        elif source_z - Agent_z < 0:
            T_VAngle = -(pi/2)
    else:  #case2:垂直角区间为(-90°, 90°)
        projection_dist = np.sqrt((Agent_x - source_x)**2 + (Agent_y - source_y)**2)
        T_VAngle = atan((source_z - Agent_z) / projection_dist)
    #理论水平角计算
    if (Agent_y - source_y == 0):  #case1:水平角=90°或-90°
        if source_x - Agent_x > 0:
            T_HAngle = pi/2
        elif source_x - Agent_x < 0:
            T_HAngle = -(pi/2)
        elif source_x - Agent_x == 0:
            T_HAngle = 0*pi
    else:  #case2:水平角区间为(-90°, 90°)
        T_HAngle = atan((source_x - Agent_x) / abs(source_y - Agent_y))
    return T_HAngle, T_VAngle   #返回值为弧度
"""
#基于人体头模型的入耳时间差计算：将[-90°, 90°]区间内以0.5°为步长的361个角度对应的入耳时间差构成用于匹配的数组
def Re_ITD_Array_Forming(N_theta):
    #N_theta = 361
    i = np.arrange(N_theta)
    Re_ITD_Array = (0.5 * sin((((i - 180) / 2) / 180) * pi)) / 340
    Re_ITD_Array = Re_ITD_Array.reshape(1, -1)
    Reference_Angle = asin(340 * Re_ITD_Array / 0.5)
    return Reference_Angle
"""

#基于人体头模型的入耳时间差计算，将[-90°, 90°]区间内以0.5°为步长的361个角度对应的入耳时间差映射为建模入耳时间差
def Re_to_M_ITD(Nh_theta, Agent_pos, source_pos):
    #N_theta = 361
    i = np.arange(Nh_theta)
    Re_ITD_Array = (0.5 * np.sin((((i - 180) / 2) / 180) * pi)) / 340
    Re_ITD_Array = Re_ITD_Array.reshape(1, -1)
    Reference_hAngle_r = np.asin(340 * Re_ITD_Array / 0.5)
    Reference_hAngle_d = np.round(Reference_hAngle_r * (180 / pi), 1)  #弧度转角度
    T_HAngle_r, _ = true_angle_cal(Agent_pos, source_pos)
    T_HAngle_d = round(T_HAngle_r * (180 / pi), 1)  #弧度转化为保留1位小数的角度
    #print(f"T_HAngle_d: {T_HAngle_d}")
    integer = math.trunc(T_HAngle_d)
    #print(f"integer: {integer}") 
    remainder = round(abs(T_HAngle_d)-abs(integer), 1)
    #print(f"remainder: {remainder}") 
    if T_HAngle_d >= 0:
        if remainder*10 >= 0 and remainder*10 <= 4:
            T_HAngle_d = integer
        elif remainder*10 >= 6 and remainder*10 <= 9:
            T_HAngle_d = integer + 1
        elif remainder*10 == 5:
            T_HAngle_d = integer + 0.5
    if T_HAngle_d < 0:
        if remainder*10 >= 0 and remainder*10 <= 4:
            T_HAngle_d = integer
        elif remainder*10 >= 6 and remainder*10 <= 9:
            T_HAngle_d = integer - 1
        elif remainder*10 == 5:
            T_HAngle_d = integer + 0.5
    #print(f"T_HAngle_d: {T_HAngle_d}")
    T_HAngle_d = round(T_HAngle_d, 1)
    index_h = np.where(T_HAngle_d == Reference_hAngle_d)
    if len(index_h[0]) > 0:
        index_h = index_h[1][0]  # 由于是二维数组，取第二维的索引
        M_ITD = Reference_hAngle_d[0, index_h] * 2  # 从二维数组中取出元素
        return round(M_ITD)
    else:
        print(f"未找到匹配的角度 {T_HAngle_d}")
        return None

def Re_to_M_ILD(Nv_theta, Agent_pos, source_pos):
    #Nv_theta = 181
    j = np.arange(Nv_theta)
    Re_ILD_Array = (j - 90) / 100
    Re_ILD_Array = Re_ILD_Array.reshape(1, -1)
    Reference_vAngle_r = ((Re_ILD_Array * 100) / 180) * pi 
    Reference_vAngle_d = np.round(Reference_vAngle_r * (180 / pi), 1)  #弧度转角度
    _, T_VAngle_r = true_angle_cal(Agent_pos, source_pos)
    T_VAngle_d = round(T_VAngle_r * (180 / pi), 1)  #弧度转化为保留1位小数的角度
    integer = math.trunc(T_VAngle_d)
    remainder = round(abs(T_VAngle_d)-abs(integer), 1) 
    if T_VAngle_d >= 0:
        if remainder*10 >= 0 and remainder*10 <= 4:
            T_VAngle_d = integer 
        elif remainder*10 >= 5 and remainder*10 <= 9:
            T_VAngle_d = integer + 1
    if T_VAngle_d < 0:
        if remainder*10 >= 0 and remainder*10 <= 4:
            T_VAngle_d = integer 
        elif remainder*10 >= 5 and remainder*10 <= 9:
            T_VAngle_d = integer = 1        
    T_VAngle_d = round(T_VAngle_d, 1)
    index_v = np.where(T_VAngle_d == Reference_vAngle_d)
    if len(index_v[0]) > 0:
        index_v = index_v[1][0]  # 由于是二维数组，取第二维的索引
        M_ILD = Reference_vAngle_d[0, index_v] / 100  # 从二维数组中取出元素
        return round(M_ILD, 2)
    else:
        print(f"未找到匹配的角度 {T_VAngle_d}")
        return None


def angle_error_cal(ec_Angle_Array, re_HAngle, re_VAngle):
    ec_HAngle = ec_Angle_Array[0]
    ec_VAngle = ec_Angle_Array[1]
    HAngle_error = np.abs(ec_HAngle - re_HAngle)
    VAngle_error = np.abs(ec_VAngle - re_VAngle)
    Angle_error = np.array([HAngle_error, VAngle_error])
    Angle_error_degree = np.rint(Angle_error * (180/pi))
    return Angle_error_degree
    
    
    
    
   
    
    
    
    
        
    
    
    

        
    
        
    
