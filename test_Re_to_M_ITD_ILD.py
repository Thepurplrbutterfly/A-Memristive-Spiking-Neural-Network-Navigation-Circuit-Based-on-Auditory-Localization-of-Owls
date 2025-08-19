import numpy as np
from math import pi
from math import sin
from math import cos
from math import asin
from math import atan
import math
def true_angle_cal_test(Agent_pos, source_pos):
    Agent_x = Agent_pos[0]
    Agent_y = Agent_pos[1]
    Agent_z = Agent_pos[2]
    source_x = source_pos[0]
    source_y = source_pos[0]
    source_z = source_pos[0]
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

def Re_to_M_ITD(Agent_pos, source_pos):
    i = np.arange(361)
    Re_ITD_Array = (0.5 * np.sin((((i - 180) / 2) / 180) * pi)) / 340
    Re_ITD_Array = Re_ITD_Array.reshape(1, -1)
    Reference_hAngle_r = np.asin(340 * Re_ITD_Array / 0.5)
    Reference_hAngle_d = np.round(Reference_hAngle_r * (180 / pi), 1)  #弧度转角度
    T_HAngle_r, _ = true_angle_cal_test(Agent_pos, source_pos)
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

def Re_to_M_ILD(Agent_pos, source_pos):
    #Nv_theta = 181
    j = np.arange(181)
    Re_ILD_Array = (j - 90) / 100
    Re_ILD_Array = Re_ILD_Array.reshape(1, -1)
    Reference_vAngle_r = ((Re_ILD_Array * 100) / 180) * pi 
    Reference_vAngle_d = np.round(Reference_vAngle_r * (180 / pi), 1)  #弧度转角度
    _, T_VAngle_r = true_angle_cal_test(Agent_pos, source_pos)
    T_VAngle_d = round(T_VAngle_r * (180 / pi), 1)  #弧度转化为保留1位小数的角度
    integer = math.trunc(T_VAngle_d)
    remainder = round(abs(T_VAngle_d)-abs(integer), 1) 
    if T_VAngle_d >= 0:
        if remainder*10 >= 0 and remainder*10 <= 4:
            T_VAngle_d = integer 
        elif remainder*10 >=5 and remainder*10 <=9:
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

def main():
    Agent_x = 12.5
    Agent_y = 10.5
    Agent_z = 25
    source_x = 45
    source_y = 45
    source_z = 45
    Agent_pos = (Agent_x, Agent_y, Agent_z)
    source_pos = (source_x, source_y, source_z)
    M_ITD = Re_to_M_ITD(Agent_pos, source_pos)
    M_ILD = Re_to_M_ILD(Agent_pos, source_pos)
    print(f"水平角 = {M_ITD / 2}")
    print(f"垂直角 = {M_ILD * 100}")

if __name__ == "__main__":
    main()