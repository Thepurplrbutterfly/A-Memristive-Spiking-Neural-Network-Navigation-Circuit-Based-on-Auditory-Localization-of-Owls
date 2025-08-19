# -*- coding:utf-8 -*-
"""
Version.CIM_Computing
"""
import numpy as np
import seaborn as sns  #数据可视化图标绘制库
import matplotlib
matplotlib.use('agg') #使用AGG作为matplotlib的后端，适用于在没有图形界面的环境(服务器)中生成图像文件
import matplotlib.pyplot as plt #导入matplotlib库中的pyplot模块，用于提供绘图函数
import matplotlib.style as mplstyle #导入matplotlib库中的mplotstyle函数，用于预定义和自定义绘图样式
mplstyle.use('fast') #设置matplotlib内置样式为fast样式，用于优化绘图速度
from tqdm.std import trange #从tqdm库中导入trange函数，tqdm是为python迭代操作添加进度条的库， trange 则是 tqdm 针对 range 函数的一个封装，使用它可以很方便地为循环添加进度显示
                             #对于一个循环，会显示一个进度条，实时展示循环的完成情况，包括已完成的百分比、剩余时间预估等信息
from time import perf_counter #用于精确测量代码执行的时间
import os #用于与操作系统交互的标准库，用于访问操作系统：(1)对文件目录操作(创建删除重命名) (2)路径处理(拼接、获取绝对路径) 等
import datetime #处理时间日期的库
from argparse import ArgumentParser #用于解析命令行参数，可通过命令行传递参数
import json #用于处理 JSON 数据，支持 JSON 格式与 Python 数据结构之间的转换
from scipy.io import savemat #scipy.io主要用于处理科学数据格式，特别是 MATLAB 的.mat文件格式。savemat函数用于将数据保存为 MATLAB 格式的文件。
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#导入忆阻阵列库
import sys #sys.argv 是一个包含命令行参数的列表，其中 sys.argv[0] 通常是脚本的名称
"""
模块搜索路径设置,使python编译器在当前和上一级目录搜索所需模块
"""
sys.path.append('./')
sys.path.append('../') 
"""
导入忆阻平台底层函数模块库
"""
#from He100_sdk.sdk_array_newsystem import SDKArray
#from He100_module import *
from AuditoryNavi_function import *
from Agent_DistanceMeasuring_noobstacle import *

"""
主函数
"""
def main():
    #-----------------创建保存输出结果的文件-----------------#
    path = './LZD_output_cpu_noobstacle/'
    path2 = 'png2_25/'
    """文件生成时间戳"""
    current_datetime = datetime.datetime.now()
    current_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")  #将时间对象转换为字符串
    current_datetime = current_datetime.replace(':', '-')  #将:转换为-，避免某些文件名不接受:
    filename = f"{path}{current_datetime}"  #将文件路径与时间生成时间戳和并作为文件名
    os.mkdir(filename)  #若已存在文件，则报错
    #-----------------创建保存输出结果的文件-----------------#
    
    #-----------------参数管理-----------------#
    parser = ArgumentParser()
    parser.add_argument('--T_v', type=int, default=100, help='垂直角检测神经元放电仿真时间')
    parser.add_argument('--T_h', type=int, default=10, help='水平角检测神经元放电仿真时间')
    parser.add_argument('--T_max', type=int, default=50000, help='最大(截止)仿真时间')
    parser.add_argument('--Num_IC', type=int, default=19, help='垂直角检测神经元数量')
    parser.add_argument('--Num_CD', type=int, default=19, help='水平角检测神经元数量')
    parser.add_argument('--Vamp', type=int, default=10, help='垂直角计算放大倍数')
    parser.add_argument('--Hamp', type=int, default=1000, help='水平角计算放大倍数')
    parser.add_argument('--repeat_times', type=int, default=1, help='重复训练次数')
    parser.add_argument('--Trial', type=int, default=64, help='单次训练次数') 
    parser.add_argument('--if_noise', action='store_true', default=False, help='是否加入权重噪声(默认不添加噪声, 启用参数则添加噪声)') 
    """命令行终端不添加参数--if_noise时则默认不添加噪声, 若添加参数--if_quanti时则添加噪声"""
    parser.add_argument('--noise_level', type=float, default=0.2, help='权重噪声水平')
    parser.add_argument('--if_quanti', action='store_false', default=True, help='是否量化权重(默认启用量化, 启用参数则禁止量化)')
    """命令行终端不添加参数--if_quanti时则默认开启量化, 若添加参数--if_quanti时则禁用量化"""
    parser.add_argument('--counter', type=int, default=1, help='如果量化权重,累计counter次后再取整')
    parser.add_argument('--if_datasave', action='store_false', default=True, help='是否保存数据(默认保存数据, 启用参数不保存)')
    """命令行终端不添加参数--if_datasave时则默认存储数据, 若添加参数--if_datasave时则不存储数据"""
    parser.add_argument('--CD_field', type=float, default=3.88, help='水平角重合检测神经元放电野')
    parser.add_argument('--Mod_var', type=float, default=6, help='修正项方差')
    parser.add_argument('--source_target', type=list, default=[15.5, 17.5, 9.5], help='默认声源位置')
    parser.add_argument('--space_size', type=list, default=[25, 25, 25], help='三维空间大小')
    parser.add_argument('--c0', type=float, default=0.2, help='权重初始化赋1比重')
    parser.add_argument('--dt_step', type=int, default=1, help='时间步长')
    parser.add_argument('--m_step', type=int, default=2, help='Agent移动步长')
    parser.add_argument('--gamma_min', type=float, default=0.19,help='最小学习率')
    parser.add_argument('--gamma_max', type=float, default=0.55,help='最大学习率')
    #W_update_N = 1  #权重更新初始次数
    args = parser.parse_args()  #读取命令行传入的参数
    
    with open(f'{filename}/commander_args.txt', 'w') as f:   #只写方式打开(若不存在则先创建)commandler_args.txt文件
        json.dump(args.__dict__, f, indent=2)   #将从命令行输入的参数转换为字典并写入commander_args.txt文件
        print('输出结果保存在',filename,'路径下')
    #-----------------参数管理-----------------#    
    
    #-----------------关键参数-----------------# 
    T_v = args.T_v   
    T_h = args.T_h 
    T_max = args.T_max
    Num_IC = args.Num_IC
    Num_CD = args.Num_CD
    Vamp = args.Vamp
    Hamp = args.Hamp      
    repeat_times = args.repeat_times
    Trial = args.Trial
    if_noise = args.if_noise
    noise_level = args.noise_level
    if_quanti = args.if_quanti
    counter = args.counter
    if_datasave = args.if_datasave
    CD_field = args.CD_field
    Mod_var = args.Mod_var
    source_target = args.source_target
    space_size = args.space_size
    c0 = args.c0
    dt_step = args.dt_step
    m_step = args.m_step
    gamma_min = args.gamma_min
    gamma_max = args.gamma_max
    update_flag = 0  #权重更新标志
    weights_update_times = 0  #权重更新次数统计
    #设置4个起始点
    _4starting_point = {0:[1,1,1],1:[space_size[0]-1,1,1],2:[1,1,space_size[2]-1],3:[space_size[0]-1,1,space_size[2]-1],4:[1,space_size[1]-1,1]}
    #-----------------关键参数-----------------# 
    
    #-----------------数据存储嵌套字典创建-----------------#
    data = {}  #创建一个空字典
    data['X_h'] = {}  #用于存储水平角神经元放电结果
    data['X_v'] = {}  #用于存储垂直角神经元放电结果
    data['W_h'] = {}  #在字典内创建一个键值为W_h的嵌套空字典，用于存储水平角神经元间权重
    data['W_v'] = {}  #用于存储垂直角神经元间权重
    data['H_Angle'] = {}  #用于存储水平角
    data['V_Angle'] = {}  #用于存储垂直角
    data['Agent_position'] = {}  #用于Agent位置
    data['x_Agent'] = {}  #用于Agent_x坐标
    data['y_Agent'] = {}  #用于Agent_y坐标
    data['z_Agent'] = {}  #用于Agent_z坐标
    data['Angle_error'] = {}  #用于估计角度与实际角度的偏差位置
    data['esti_distance-true_distance'] = {}  #用于存储距离
    #-----------------数据存储嵌套字典创建-----------------#
    
    #初始化水平角权重
    W_h = np.random.uniform(0,1,(Num_CD,Num_CD))
    W_v = np.random.uniform(0,1,(Num_IC,Num_IC))
    W_h_original = W_h
    W_v_original = W_v
    data['W_h_original'] = W_h_original  #创建一个存储水平角初始权重矩阵的字典
    np.savetxt(f"./{filename}/W_h_original.txt", W_h_original)  #将初始水平角权重存储在W_h_original.txt文件中
    data['W_v_original'] = W_v_original  #创建一个存储垂直角初始权重矩阵的字典
    np.savetxt(f"./{filename}/W_v_original.txt", W_v_original)  #将初始垂直角权重存储在W_v_original.txt文件中
    
    if args.if_quanti:
        #==============将生成的[0, 1)区间内的初始权重量化至区间[-8, 7]==============#
        W_h = (((W_h - np.min(W_h)) / (np.max(W_h) - np.min(W_h))) * 15 - 8).round()
        W_v = (((W_v - np.min(W_v)) / (np.max(W_v) - np.min(W_v))) * 15 - 8).round()
        #==============将生成的[0, 1)区间内的初始权重量化至区间[-8, 7]==============#
        W_h = W_h + 8
        W_v = W_v + 8
        print(np.max(W_h), np.min(W_h), np.max(W_v), np.min(W_v))
        W_h_ori_q = (((W_h_original - np.min(W_h_original)) / (np.max(W_h_original) - np.min(W_h_original))) * 15 - 8).round()
        W_v_ori_q = (((W_v_original - np.min(W_v_original)) / (np.max(W_v_original) - np.min(W_v_original))) * 15 - 8).round()
    data['W_h_ori_q'] = W_h_ori_q  #创建一个存储水平角量化权重矩阵的字典
    data['W_v_ori_q'] = W_v_ori_q  #创建一个存储水平角量化权重矩阵的字典
    np.savetxt(f"./{filename}/W_h_ori_quanti.txt", W_h_ori_q)  #将初始水平角量化后的权重存储在W_h_or_quanti.txt文件中 
    np.savetxt(f"./{filename}/W_v_ori_quanti.txt", W_v_ori_q)  #将初始垂直角量化后的权重存储在W_v_or_quanti.txt文件中 
    """
    #-----------------写水平角权重-----------------#
    weight_h_wr_chip = W_h
    [row, col] = weight_h_wr_chip.shape
    weight_h_wr_chip_adder = [0, 0, row, col]
    weight_h_adder = weight_h_wr_chip_adder
    sdk = SDKArray(0)
    mapping_repeat = 1
    for _ in range(mapping_repeat):
        sdk.set_weight(weight = weight_h_wr_chip + 8, adder = weight_h_wr_chip_adder, verbose = 0)  #将量化后的[-8, 7]区间内的权值+8后映射至忆阻平台阻值[0, 15]
    weight_h_4bit_on_chip = sdk.get_weight(weight_h_adder, verbose = 0)
    data['weight_h_4bit_on_chip_ini'] = weight_h_4bit_on_chip  #创建一个存储忆阻平台对应水平角初始权重的字典
    np.savetxt(f"./{filename}/weight_h_4bit_on_chip_ini.txt")  #将忆阻平台对应水平角权重存储在W_h_ori_memc文件中
    #-----------------写水平角权重-----------------#
    
    #-----------------写垂直角权重-----------------#
    weight_v_wr_chip = W_v
    [row2, col2] = weight_v_wr_chip.shape
    weight_v_wr_chip_adder = [25, 25, row2, col2]
    weight_v_adder = weight_v_wr_chip_adder
    mapping_repeat1 = 1
    for _ in range(mapping_repeat1):
        sdk.set_weight(weight = weight_v_wr_chip + 8, adder = weight_v_wr_chip_adder, verbose = 0)
    weight_v_4bit_on_chip = sdk.get_weight(weight_v_adder, verbose = 0)
    data['weight_v_4bit_on_chip_ini'] = weight_v_4bit_on_chip  #创建一个存储忆阻平台对应垂直角初始权重的字典
    np.savetxt(f"./{filename}/weight_v_4bit_on_chip_ini.txt")  #将忆阻平台对应垂直角权重存储在W_v_ori_memc文件中
    #-----------------写垂直角权重-----------------#
    """
    #-----------------权重矩阵图绘制-----------------#
    cmap1 = LinearSegmentedColormap.from_list('_3_colors', ['mediumspringgreen', 'lavender','b'])
    cmap2 = LinearSegmentedColormap.from_list('W_on_chip_colors', ['palevioletred', 'thistle', 'azure', '#B3B3FF', 'navy'])
    cmap3 = LinearSegmentedColormap.from_list('_6_colors', ['springgreen', 'mediumseagreen', 'c', 'dodgerblue', 'royalblue' ,'b'])
    cmap5 = LinearSegmentedColormap.from_list('CD_neurons_ini_W_colors', ['ghostwhite', 'lavender', 'lightskyblue', 'skyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'blue', 'mediumblue', 'navy'])
    cmap6 = LinearSegmentedColormap.from_list('IC_neurons_ini_W_colors', ['#FEF9CB', '#FDF3A2', '#FCEE74', '#FBE731','#FBD831', '#FBD006',  'orange', 'darkorange', '#F16D1F', '#ED5A14'])
    cmap7 = LinearSegmentedColormap.from_list('CD_neurons_firing_rate_colors', ['lavender', 'dodgerblue', 'blue'])
    cmap8 = LinearSegmentedColormap.from_list('IC_neurons_firing_rate_colors', ['papayawhip', 'gold', 'orange'])
    cmap9 = LinearSegmentedColormap.from_list('CD_neurons_firing_colors', ['lavender', 'blue'])
    cmap10 = LinearSegmentedColormap.from_list('IC_neurons_firing_colors', ['papayawhip', 'orange'])
# 绘制热力图
    fig_W = plt.figure()
    plt.subplot(2, 2, 1)
    sns.heatmap(W_h_ori_q, annot=False, cmap=cmap5, xticklabels='auto', yticklabels='auto', linewidths=0)
    plt.title('Horizontal unquantized weights')
    
    plt.subplot(2, 2, 2)
    sns.heatmap(W_v_ori_q, annot=False, cmap=cmap6, xticklabels='auto', yticklabels='auto', linewidths=0)
    plt.title('Vertical unquantized weights')
    
    plt.subplot(2, 2, 3)
    sns.heatmap(W_h, annot=False, cmap=cmap5, xticklabels='auto', yticklabels='auto', linewidths=0)
    plt.title('Horizontal quantized weights')
    
    plt.subplot(2, 2, 4)
    sns.heatmap(W_v, annot=False, cmap=cmap6, xticklabels='auto', yticklabels='auto', linewidths=0)
    plt.title('Vertical quantized weights')
    """
    plt.subplot(3, 2, 3)
    sns.heatmap(weight_h_4bit_on_chip, annot=False, cmap=cmap2, xticklabels='auto', yticklabels='auto', linewidths=0)
    plt.title('Horizontal weights on chip')
    
    plt.subplot(2, 2, 6)
    sns.heatmap(weight_v_4bit_on_chip, annot=False, cmap=cmap2, xticklabels='auto', yticklabels='auto', linewidths=0)
    plt.title('Vertical weights on chip')
    """
    fig_W.tight_layout()  #自动调整子图间距
    plt.savefig(f"./{filename}/Quantized_Onchip_Weights_Figure", dpi=220)
    plt.close()
    #-----------------权重矩阵图绘制-----------------#
    
    #-----------------遍历Trial次实验-----------------#
    Trial = args.Trial
    repeat_times = args.repeat_times
    #results = np.zeros((Trial, repeat_times))
    #results_per_dis = np.zeros((Trial, repeat_times))
    #angle_errors = np.zeros((Trial, repeat_times))
    start_time = perf_counter()  #开始计时
    for R in range(repeat_times):
        #==============再次创建一个嵌套字典用于存储第R次重复实验的数据==============#
        data['X_h'][f'{R}-th'] = {}
        data['X_v'][f'{R}-th'] = {}
        data['W_h'][f'{R}-th'] = {}  
        data['W_v'][f'{R}-th'] = {}  
        data['H_Angle'][f'{R}-th'] = {}  
        data['V_Angle'][f'{R}-th'] = {}  
        data['Agent_position'][f'{R}-th'] = {}  
        data['x_Agent'][f'{R}-th'] = {}
        data['y_Agent'][f'{R}-th'] = {}
        data['z_Agent'][f'{R}-th'] = {}
        data['Angle_error'][f'{R}-th'] = {}  
        data['esti_distance-true_distance'][f'{R}-th'] = {}  
        #W_ini_h = W_h
        #W_ini_v = W_v 
        env = Auditory_Environment(space_size = space_size)
        decay = 1   #衰减因子
        for Tr in range(Trial):  #每次重复实验中对Trial次独立实验进行遍历
            #==============Agent初始位置坐标==============#
            x_ini, y_ini, z_ini = _4starting_point[Tr % 4]
            #x_ini, y_ini, z_ini = [1, 1, 1]
            x_Agent = x_ini
            y_Agent = y_ini
            z_Agent = z_ini  
            new_position = [x_Agent, y_Agent, z_Agent]
            #==============Agent初始位置坐标==============#
            arr_flag = 0   #到达声源3.5米半径内标志
            n = 1  #步长次数记录
            #==============Agent声源前后方判断传感器位置初始化==============#
            front_sensor_pos = [x_Agent, y_Agent + 1, z_Agent] 
            back_sensor_pos = [x_Agent, y_Agent - 1, z_Agent]
            #==============Agent声源前后方判断传感器位置初始化==============#
            
            perce = Auditory_Perception(initial_pos = (x_Agent, y_Agent, z_Agent), env = env)
            esti_dist, true_dist, _ = perce.dist_estimating(env.sound_source(), source_target)
            new_distacne = [esti_dist, true_dist]
            print(f'第{R+1}次重复实验中的第{Tr+1}次独立实验:Agent起始位置为{[x_Agent, y_Agent, z_Agent]}, Agent与声源的实际距离为{round(true_dist, 2)}, 估计距离为{round(esti_dist, 2)}')
            #==============创建历史轨迹数组以存储Agent运动坐标点==============#
            x_Agent_history = []
            y_Agent_history = []
            z_Agent_history = []
            Agent_pos_history = []
            encoded_HAngle_history = []
            encoded_VAngle_history = []
            true_dist_history = []
            esti_dist_history = []
            x_Agent_history.append(x_Agent)
            y_Agent_history.append(y_Agent)
            z_Agent_history.append(z_Agent)
            Agent_pos_history.append(np.array(new_position))
            encoded_HAngle_history.append(0)
            encoded_VAngle_history.append(0)
            true_dist_history.append(round(true_dist, 2))
            esti_dist_history.append(round(esti_dist, 2))
            data['X_h'][f'{R}-th'][f'{Tr}-th'] = np.zeros((int(T_max/T_v), 19, 20)) 
            data['X_v'][f'{R}-th'][f'{Tr}-th'] = np.zeros((int(T_max/T_v), 19, 100)) 
            data['W_h'][f'{R}-th'][f'{Tr}-th'] = np.zeros((int(T_max/T_v), 19, 19))  
            data['W_v'][f'{R}-th'][f'{Tr}-th'] = np.zeros((int(T_max/T_v), 19, 19))  
            data['H_Angle'][f'{R}-th'][f'{Tr}-th'] = np.zeros(int(T_max/T_v))  
            data['V_Angle'][f'{R}-th'][f'{Tr}-th'] = np.zeros(int(T_max/T_v))  
            data['Agent_position'][f'{R}-th'][f'{Tr}-th'] = np.zeros((int(T_max/T_v), 3))
            data['x_Agent'][f'{R}-th'][f'{Tr}-th'] = np.zeros(int(T_max/T_v))
            data['y_Agent'][f'{R}-th'][f'{Tr}-th'] = np.zeros(int(T_max/T_v))
            data['z_Agent'][f'{R}-th'][f'{Tr}-th'] = np.zeros(int(T_max/T_v))
            data['Angle_error'][f'{R}-th'][f'{Tr}-th'] = np.zeros((int(T_max/T_v), 2))
            data['esti_distance-true_distance'][f'{R}-th'][f'{Tr}-th'] = np.zeros((int(T_max/T_v), 2))
            #==============遍历T_max/T_v步长==============#
            for t in trange(int(T_max/T_v), desc=f'第{R+1}次重复实验中的第{Tr+1}次独立实验'):
                """trange用于创建一个进度条显示迭代器, 显示步长执行进度:i/T_max"""
                #==============使用CPU计算水平角CD神经元和垂直角IC神经元的输出=============# 
                M_ITD = Re_to_M_ITD(Nh_theta = 361, Agent_pos = (x_Agent, y_Agent, z_Agent), source_pos = source_target)
                vh_rate = Hn_FIT(M_ITD, Num_CD, Hamp, CD_field, Mod_var)  #19个水平角检测神经元放电强度, vh_rate为数组
                h_fire = Hn_Fire(T_h, vh_rate)
                X_h = h_fire  #T_h内19个神经元的放电次数数组
                t_h = np.arange(1, T_h*2+1, dt_step)
                HCD_CPU_Output = np.zeros((Num_CD, len(t_h)))
                """
                for t_ih in t_h:
                    H_input_array = X_h[:, 1, t_ih-1].reshape(-1, 1)  #将19个水平角神经元在一个时间步t_ih-1时刻的放电情况存储为一个(19, 1)大小的矩阵
                    #==============计算一个时间步的加权和=============#
                    HCD_MVM_rram = linear_144k(sdk, input_feature_map = H_input_array, 
                                           weight_addr = weight_h_adder, repeat = [1, 1], 
                                           input_half_level = 127, output_half_level = 127, 
                                           it_time = 5, 
                                           relu = False)
                    HCD_MVM_Output = HCD_MVM_rram
                """
                M_ILD = Re_to_M_ILD(Nv_theta = 181, Agent_pos = (x_Agent, y_Agent, z_Agent), source_pos = source_target)   
                vv_rate = Vn_FIT(M_ILD, Num_IC, Vamp)  #19个垂直角检测神经元放电强度, vv_rate为数组 
                v_fire = Vn_Fire(T_v, vv_rate)
                X_v = v_fire  #T_v内19个神经元的放电次数数组
                t_v = np.arange(1, T_v+1, dt_step)
                VIC_CPU_Output = np.zeros((Num_IC, len(t_v)))
                t_ih = 1
                for t_iv in t_v:
                    """由于CD神经元输出计算单步长时间仅为IC神经元输出计算单步长时间的1/10, 所以这里在IC神经元输出计算的前10次遍历的同时对IC神经元输出进行计算"""
                    if t_ih <= np.max(t_h):
                        H_input_array = X_h[:, 1, t_ih-1].reshape(-1, 1)  #将19个水平角神经元在一个时间步t_ih-1时刻的放电情况存储为一个(19, 1)大小的矩阵
                        #==============计算水平角神经元在一个单时间步的加权和=============#
                        HCD_CPU = np.dot(W_h.T, H_input_array)
                        #print(f"HCD_CPU 形状: {HCD_CPU.shape}") 
                        #t_ih += 1
                        HCD_CPU_Output[:, t_ih-1] = np.squeeze(HCD_CPU)
                        #print(f"HCD_CPU_Output形状: {HCD_CPU_Output[:, t_ih-1].shape}") 
                        t_ih += 1
                    V_input_array = X_v[:, 1, t_iv-1].reshape(-1, 1)  #将19个垂直角神经元在一个时间步t_ih-1时刻的放电情况存储为一个(19, 1)大小的矩阵
                    #==============计算垂直角神经元在一个单时间步的加权和=============#
                    VIC_CPU = np.dot(W_v.T, V_input_array)
                    VIC_CPU_Output[:, t_iv-1] = np.squeeze(VIC_CPU)
                HCD_CPU_Output_sum = np.zeros((Num_CD, 1))
                VIC_CPU_Output_sum = np.zeros((Num_IC, 1))
                f_sum_Hn = np.zeros((Num_CD, 1))
                f_sum_Vn = np.zeros((Num_IC, 1))
                X_h_temp = X_h
                #print(f"f_sum_Hn 形状: {f_sum_Hn.shape}") 
                X_v_temp = X_v
                tt_ih = 1
                #HCD_MVM_Output_sum_rec = np.zeros((T_max/T_v, Num_CD))
                #VIC_MVM_Output_sum_rec = np.zeros((T_max/T_v, Num_IC))
                """对一个步长内每个单步长的计算结果极性累加"""
                for tt_iv in t_v:
                    #==============对水平角神经元在一个时间步内的输入和输出进行累加=============#
                    if tt_ih <= np.max(t_h):
                        f_sum_Hn = X_h_temp[:, 1, tt_ih-1].reshape(-1, 1) + f_sum_Hn
                        #print(f"f_sum_Hn 形状: {f_sum_Hn.shape}") 
                        HCD_CPU_Output_sum = HCD_CPU_Output[:, tt_ih-1].reshape(-1, 1) + HCD_CPU_Output_sum
                        #print(f"HCD_CPU_Output_sum 形状: {HCD_CPU_Output_sum.shape}") 
                        tt_ih += 1
                    #==============对垂直角角神经元在一个时间步内的输入和输出进行累加=============#
                    f_sum_Vn = X_v_temp[:, 1, tt_iv-1].reshape(-1, 1) + f_sum_Vn
                    VIC_CPU_Output_sum = VIC_CPU_Output[:, tt_iv-1].reshape(-1, 1) + VIC_CPU_Output_sum
                HCD_CPU_Output_sum = HCD_CPU_Output_sum.reshape(1, -1)
                VIC_CPU_Output_sum = VIC_CPU_Output_sum.reshape(1, -1)
                f_sum_Hn = f_sum_Hn.reshape(1, -1)
                f_sum_Vn = f_sum_Vn.reshape(1, -1)
                #==============神经元群体编码=============#    
                encoded_Angles = population_encoding(HCD_CPU_Output_sum, VIC_CPU_Output_sum)
                #编码后水平角度
                encoded_HAngle = encoded_Angles[0]
                data['H_Angle'][f'{R}-th'][f'{Tr}-th'][t] = encoded_HAngle 
                encoded_HAngle_history.append(encoded_HAngle)
                #编码后垂直角度
                encoded_VAngle = encoded_Angles[1]
                data['V_Angle'][f'{R}-th'][f'{Tr}-th'][t] = encoded_VAngle
                encoded_VAngle_history.append(encoded_VAngle)
                
                #==============角度误差计算=============#  
                T_HAngle, T_VAngle = true_angle_cal(np.array(new_position), source_target)
                Angle_error_degree = angle_error_cal(encoded_Angles, T_HAngle, T_VAngle)
                
                #==============胜者神经元标志生成=============#
                H_winner_Flag_Array = np.zeros((1, Num_CD))
                V_winner_Flag_Array = np.zeros((1, Num_IC))
                WTA_flag = WTA_flag_update(HCD_CPU_Output_sum, VIC_CPU_Output_sum)
                #水平角神经元胜者标志生成：胜者置1，输者置0
                H_winner_Flag_Array = WTA_flag[0, :]
                #垂直角神经元胜者标志生成：胜者置1，输者置0
                V_winner_Flag_Array = WTA_flag[1, :]
                
                #==============Agent按编码角移动=============#
                theta_h = encoded_HAngle
                theta_v = encoded_VAngle
                """
                data['position'][f'{R}-th'][f'{Tr}-th'][-1] = np.array(new_position)
                data['x_Agent'][f'{R}-th'][f'{Tr}-th'][-1] = x_Agent
                data['y_Agent'][f'{R}-th'][f'{Tr}-th'][-1] = y_Agent
                data['z_Agent'][f'{R}-th'][f'{Tr}-th'][-1] = z_Agent
                data['esti_distance-true_distance'][f'{R}-th'][f'{Tr}-th'][-1] = np.array(new_distacne)
                """
                x_old = x_Agent
                y_old = y_Agent
                z_old = z_Agent
                front_sensor_pos_old = front_sensor_pos
                back_sensor_pos_old = back_sensor_pos
                """当距离大于3.5米时进行移动"""
                if esti_dist > 1.5: 
                    """当声源位于Agent左右两侧时"""
                    if theta_h == pi/2 or theta_h == -pi/2:
                        #--------Agent坐标更新--------#
                        x_Agent = x_old + (m_step * cos(theta_v)) * sin(theta_h)
                        y_Agent = y_old + (m_step * cos(theta_v)) * cos(theta_h)
                        z_Agent = z_old + m_step * sin(theta_v)
                        #--------前置传感器坐标更新--------#
                        front_sensor_pos[0] = x_Agent
                        front_sensor_pos[1] = front_sensor_pos_old[1] + (m_step * cos(theta_v)) * cos(theta_h)
                        front_sensor_pos[2] = z_Agent
                        #--------后置传感器坐标更新--------#
                        back_sensor_pos[0] = x_Agent
                        back_sensor_pos[1] = back_sensor_pos_old[1] + (m_step * cos(theta_v)) * cos(theta_h)
                        back_sensor_pos[2] = z_Agent
                        Reward_value = 0
                        
                        #当Agent位置更新后超出边界时：
                        if x_Agent > space_size[0]-1:
                            x_Agent = space_size[0]-1 - 1
                            front_sensor_pos[0] = x_Agent
                            back_sensor_pos[0] = x_Agent
                            Reward_value = -1
                        if y_Agent > space_size[1]-1:
                            y_Agent = space_size[1]-1 - 1
                            front_sensor_pos[1] = y_Agent + 1
                            back_sensor_pos[1] = y_Agent - 1
                            Reward_value = -1
                        if z_Agent > space_size[2]-1:
                            z_Agent = space_size[2]-1 - 1
                            front_sensor_pos[2] = z_Agent 
                            back_sensor_pos[2] = z_Agent 
                            Reward_value = -1
                        if x_Agent < 1:
                            x_Agent = 1 + 1
                            front_sensor_pos[0] = x_Agent
                            back_sensor_pos[0] = x_Agent
                            Reward_value = -1
                        if y_Agent < 1:
                            y_Agent = 1 + 1
                            front_sensor_pos[1] = y_Agent + 1
                            back_sensor_pos[1] = y_Agent - 1
                            Reward_value = -1
                        if z_Agent < 1:
                            z_Agent = 1 + 1
                            front_sensor_pos[2] = z_Agent
                            back_sensor_pos[2] = z_Agent
                            Reward_value = -1
                            
                        
                    else:
                        """当声源位于Agent前方时"""
                        if round(np.linalg.norm(np.array(source_target) - np.array(front_sensor_pos)), 10) < round(np.linalg.norm(np.array(source_target) - np.array(back_sensor_pos)), 10):
                        ##if (source_target[1] - front_sensor_pos[1]) > (source_target[1] - back_sensor_pos[1]):   
                            #--------Agent坐标更新--------#
                            x_Agent = x_old + (m_step * cos(theta_v)) * sin(theta_h)
                            y_Agent = y_old + (m_step * cos(theta_v)) * cos(theta_h)
                            z_Agent = z_old + m_step * sin(theta_v)
                            #--------前置传感器坐标更新--------#
                            front_sensor_pos[0] = x_Agent
                            front_sensor_pos[1] = front_sensor_pos_old[1] + (m_step * cos(theta_v)) * cos(theta_h)
                            front_sensor_pos[2] = z_Agent
                            #--------后置传感器坐标更新--------#
                            back_sensor_pos[0] = x_Agent
                            back_sensor_pos[1] = back_sensor_pos_old[1] + (m_step * cos(theta_v)) * cos(theta_h)
                            back_sensor_pos[2] = z_Agent
                            Reward_value = 0
                            
                        #当Agent位置更新后超出边界时：
                        if x_Agent > space_size[0]-1:
                            x_Agent = space_size[0]-1 - 1
                            front_sensor_pos[0] = x_Agent
                            back_sensor_pos[0] = x_Agent
                            Reward_value = -1
                        if y_Agent > space_size[1]-1:
                            y_Agent = space_size[1]-1 - 1
                            front_sensor_pos[1] = y_Agent + 1
                            back_sensor_pos[1] = y_Agent - 1
                            Reward_value = -1
                        if z_Agent > space_size[2]-1:
                            z_Agent = space_size[2]-1 - 1
                            front_sensor_pos[2] = z_Agent 
                            back_sensor_pos[2] = z_Agent 
                            Reward_value = -1
                        if x_Agent < 1:
                            x_Agent = 1 + 1
                            front_sensor_pos[0] = x_Agent
                            back_sensor_pos[0] = x_Agent
                            Reward_value = -1
                        if y_Agent < 1:
                            y_Agent = 1 + 1
                            front_sensor_pos[1] = y_Agent + 1
                            back_sensor_pos[1] = y_Agent - 1
                            Reward_value = -1
                        if z_Agent < 1:
                            z_Agent = 1 + 1
                            front_sensor_pos[2] = z_Agent
                            back_sensor_pos[2] = z_Agent
                            Reward_value = -1
                            
                        """当声源位于Agent后方时"""
                        if round(np.linalg.norm(np.array(source_target) - np.array(front_sensor_pos)), 10) > round(np.linalg.norm(np.array(source_target) - np.array(back_sensor_pos)), 10):
                        ##if (source_target[1] - front_sensor_pos[1]) < (source_target[1] - back_sensor_pos[1]):
                            #--------Agent坐标更新--------#
                            x_Agent = x_old + (m_step * cos(theta_v)) * sin(pi - theta_h)
                            y_Agent = y_old + (m_step * cos(theta_v)) * cos(pi - theta_h)
                            z_Agent = z_old + m_step * sin(theta_v)
                            #--------前置传感器坐标更新--------#
                            front_sensor_pos[0] = x_Agent
                            front_sensor_pos[1] = front_sensor_pos_old[1] + (m_step * cos(theta_v)) * cos(pi - theta_h)
                            front_sensor_pos[2] = z_Agent
                            #--------后置传感器坐标更新--------#
                            back_sensor_pos[0] = x_Agent
                            back_sensor_pos[1] = back_sensor_pos_old[1] + (m_step * cos(theta_v)) * cos(pi - theta_h)
                            back_sensor_pos[2] = z_Agent
                            Reward_value = 0
                        #当Agent位置更新后超出边界时：
                        if x_Agent > space_size[0]-1:
                            x_Agent = space_size[0]-1 - 1
                            front_sensor_pos[0] = x_Agent
                            back_sensor_pos[0] = x_Agent
                            Reward_value = -1
                        if y_Agent > space_size[1]-1:
                            y_Agent = space_size[1]-1 - 1
                            front_sensor_pos[1] = y_Agent + 1
                            back_sensor_pos[1] = y_Agent - 1
                            Reward_value = -1
                        if z_Agent > space_size[2]-1:
                            z_Agent = space_size[2]-1 - 1
                            front_sensor_pos[2] = z_Agent 
                            back_sensor_pos[2] = z_Agent 
                            Reward_value = -1
                        if x_Agent < 1:
                            x_Agent = 1 + 1
                            front_sensor_pos[0] = x_Agent
                            back_sensor_pos[0] = x_Agent
                            Reward_value = -1
                        if y_Agent < 1:
                            y_Agent = 1 + 1
                            front_sensor_pos[1] = y_Agent + 1
                            back_sensor_pos[1] = y_Agent - 1
                            Reward_value = -1
                        if z_Agent < 1:
                            z_Agent = 1 + 1
                            front_sensor_pos[2] = z_Agent
                            back_sensor_pos[2] = z_Agent
                            Reward_value = -1
                            
                """
                elif esti_dist <= 3.5 and esti_dist >= 0:
                    #arr_flag = 1
                    x_Agent = x_old
                    y_Agent = y_old
                    z_Agent = z_old
                    front_sensor_pos[0] = front_sensor_pos_old[0]
                    front_sensor_pos[1] = front_sensor_pos_old[1]
                    front_sensor_pos[2] = front_sensor_pos_old[2]    
                    #Reward_value = 1     
                """          
                x_Agent_history.append(x_Agent)
                y_Agent_history.append(y_Agent)
                z_Agent_history.append(z_Agent)
                new_position = [x_Agent, y_Agent, z_Agent]
                Agent_pos_history.append(np.array(new_position))
                
                #==============计算移动后Agent对声源距离的估计=============#
                perce.Agent_pos = new_position
                esti_dist, true_dist, _ = perce.dist_estimating(env.sound_source(), source_target)
                true_dist_history.append(round(true_dist, 2))
                esti_dist_history.append(round(esti_dist, 2))
                new_distacne = [esti_dist, true_dist]
                
                #==============基于声距的奖惩值确定=============#
                n_esti_dist = esti_dist_history[-1]
                o_esti_dist = esti_dist_history[-2]
                if n_esti_dist > 1.5:
                    if n_esti_dist <= o_esti_dist:
                        Reward_value = 1
                    elif n_esti_dist > o_esti_dist:
                        Reward_value = -1
                elif n_esti_dist >= 0 and n_esti_dist <= 1.5:
                    arr_flag = 1
                    Reward_value = 0
                    
                #==============学习率调整=============#
                delta_dis = abs(n_esti_dist - o_esti_dist)
                
                if n_esti_dist <= o_esti_dist:
                    gamma = gamma_min + (gamma_max - gamma_min) * (1 / (1 + delta_dis))
                elif n_esti_dist > o_esti_dist:
                    gamma = gamma_min + (gamma_max - gamma_min) * (delta_dis / (1 + delta_dis))
                
                gamma = gamma_min + (gamma_max - gamma_min) * (abs(n_esti_dist-3.5) / (1 + abs(n_esti_dist-3.5)))
                    
                #==============权重更新=============#
                #var_W = np.ones((Num_CD, Num_CD), dtype = int)
                if Reward_value != 0:
                    update_flag = 1
                    ## 水平角神经元权重更新 ##
                    delta_W_h = np.ones_like(W_h, dtype = int)
                    #将水平角赢家标志数组按行复制19次
                    H_winner_Flag_Array_copy = np.tile(H_winner_Flag_Array, (19, 1))
                    #获取19个水平角神经元放电标志数组并按列复制19次
                    f_sum_Hn_fire_flag = ((f_sum_Hn > 0.01).astype(int)).reshape(-1, 1)
                    num1_H = np.sum(f_sum_Hn_fire_flag == 1)
                    if num1_H == 2:
                        index1_H = np.where(f_sum_Hn_fire_flag == 1)[0]   #取行索引:此时f_sum_Hn_fire_flag为一维数组，所以取唯一的一行，如果是二维，则还需要取列索引
                        _1st_index1_H = index1_H[0]
                        _2nd_index1_H = index1_H[1]
                        if _1st_index1_H == 0:
                            f_sum_Hn_fire_flag[_2nd_index1_H + 1] = 1
                        elif _2nd_index1_H == len(f_sum_Hn_fire_flag) - 1:
                            f_sum_Hn_fire_flag[_1st_index1_H - 1] = 1
                        else:
                            f_sum_Hn_fire_flag[_1st_index1_H - 1] = 1
                            f_sum_Hn_fire_flag[_2nd_index1_H + 1] = 1
                    elif num1_H == 1:
                        index1_H = np.where(f_sum_Hn_fire_flag == 1)[0]
                        if index1_H == 0:
                            f_sum_Hn_fire_flag[index1_H + 1] = 1
                        elif index1_H == len(f_sum_Hn_fire_flag) - 1:
                            f_sum_Hn_fire_flag[index1_H - 1] = 1
                        else:
                            f_sum_Hn_fire_flag[index1_H - 1] = 1
                            f_sum_Hn_fire_flag[index1_H + 1] = 1           
                    f_sum_Hn_fire_flag_copy = np.tile(f_sum_Hn_fire_flag, (1, 19))
                    #权重更新值确定
                    delta_W_h = delta_W_h * H_winner_Flag_Array_copy * f_sum_Hn_fire_flag_copy
                    W_h_new = W_h + delta_W_h * Reward_value * gamma * decay
                    #W_h_new = np.where(W_h_new < 0, 0, np.where(W_h_new > 15, 15, W_h_new)) #将更新后权重限制在[0,15]
                    
                    ## 垂直角神经元权重更新 ##
                    delta_W_v = np.ones_like(W_v, dtype = int)
                    #将垂直角赢家标志数组复制19行
                    V_winner_Flag_Array_copy = np.tile(V_winner_Flag_Array, (19, 1)) 

                    #获取19个垂直角神经元放电标志数组并按列复制19次
                    f_sum_Vn_fire_flag = ((f_sum_Vn > 0.01).astype(int)).reshape(-1, 1)
                    num1_V = np.sum(f_sum_Vn_fire_flag == 1)
                    if num1_V == 2:
                        index1_V = np.where(f_sum_Vn_fire_flag == 1)[0]
                        _1st_index1_V = index1_V[0]
                        _2nd_index1_V = index1_V[1]
                        if _1st_index1_V == 0:
                            f_sum_Vn_fire_flag[_2nd_index1_V + 1] = 1
                        elif _2nd_index1_V == len(f_sum_Vn_fire_flag) - 1:
                            f_sum_Vn_fire_flag[_1st_index1_V - 1] = 1
                        else:
                            f_sum_Vn_fire_flag[_1st_index1_V - 1] = 1
                            f_sum_Vn_fire_flag[_2nd_index1_V + 1] = 1
                    elif num1_V == 1:
                        index1_V = np.where(f_sum_Vn_fire_flag == 1)[0]
                        if index1_V == 0:
                            f_sum_Vn_fire_flag[index1_V + 1] = 1
                        elif index1_V == len(f_sum_Vn_fire_flag) - 1:
                            f_sum_Vn_fire_flag[index1_V - 1] = 1
                        else:
                            f_sum_Vn_fire_flag[index1_V - 1] = 1
                            f_sum_Vn_fire_flag[index1_V + 1] = 1
                    f_sum_Vn_fire_flag_copy = np.tile(f_sum_Vn_fire_flag, (1, 19))
                    #权重更新值确定
                    delta_W_v = delta_W_v * V_winner_Flag_Array_copy * f_sum_Vn_fire_flag_copy
                    W_v_new = W_v + delta_W_v * Reward_value * gamma * decay
                    #W_v_new = np.where(W_v_new < 0, 0, np.where(W_v_new > 15, 15, W_v_new)) #将更新后权重限制在[0,15]
                    
                #权重归一化-量化-更新
                if update_flag == 1:
                    update_flag == 0
                    W_h_new = (W_h_new - np.min(W_h_new)) / (np.max(W_h_new) - np.min(W_h_new))
                    W_v_new = (W_v_new - np.min(W_v_new)) / (np.max(W_v_new) - np.min(W_v_new))
                    if args.if_quanti:
                        W_h_new = W_h_new * 15 - 8
                        W_h = W_h_new
                        W_h = W_h_new + 8
                        W_v_new = W_v_new * 15 - 8
                        W_v = W_v_new
                        W_v = W_v_new + 8
                        #weight_h_wr_chip = W_h
                        #weight_v_wr_chip = W_v
                        #mapping_repeat = 1
                    weights_update_times += 1
                    """
                    for _ in range(mapping_repeat): 
                        #写入新权值
                        sdk.set_weight(weight = weight_h_wr_chip + 8, adder = weight_h_wr_chip_adder, verbose = 0)
                        sdk.set_weight(weight = weight_v_wr_chip + 8, adder = weight_v_wr_chip_adder, verbose = 0)
                    print(f'第{weights_update_times}次片上权值更新')
                    """
                #绘图:当Agent到达声源半径3.5m内时且实验次数Tr为10的整数倍时进行绘图
                if (arr_flag == 1): #and (Tr % 10 == 0):
                    """更新数据存储"""
                    data['X_h'][f'{R}-th'][f'{Tr}-th'][t] = X_h[:, 1, :]
                    data['X_v'][f'{R}-th'][f'{Tr}-th'][t] = X_v[:, 1, :]
                    data['W_v'][f'{R}-th'][f'{Tr}-th'][t] = W_v
                    data['W_h'][f'{R}-th'][f'{Tr}-th'][t] = W_h
                    #data['Angle'][f'{R}-th'][f'{Tr}-th'][t] = np.array([theta_h, theta_v])
                    data['Agent_position'][f'{R}-th'][f'{Tr}-th'][t] = np.array(new_position)
                    data['x_Agent'][f'{R}-th'][f'{Tr}-th'][t] = x_Agent
                    data['y_Agent'][f'{R}-th'][f'{Tr}-th'][t] = y_Agent
                    data['z_Agent'][f'{R}-th'][f'{Tr}-th'][t] = z_Agent
                    data['Angle_error'][f'{R}-th'][f'{Tr}-th'][t] = Angle_error_degree
                    data['esti_distance-true_distance'][f'{R}-th'][f'{Tr}-th'][t] = np.array(new_distacne)
                    """    
                    #==============新片上权重获取=============#
                    weight_h_4bit_on_chip_new = sdk.get_weight(weight_h_adder, verbose = 0)  
                    weight_v_4bit_on_chip_new = sdk.get_weight(weight_v_adder, verbose = 0)  
                    data['weight_h_4bit_on_chip_new'][f'{R}-th'][f'{Tr}-th'][t] = weight_h_4bit_on_chip_new
                    data['weight_v_4bit_on_chip_new'][f'{R}-th'][f'{Tr}-th'][t] = weight_v_4bit_on_chip_new
                    """
                    """绘图开始"""    
                    fig_all = plt.figure(figsize=(8, 9))
                    #==============在图形窗口中创建一个子图网络=============#
                    #subplt_grid = plt.GridSpec(4, 6, wspace=0.6, hspace=0.6)
                    #--------------三维空间中Agent移动轨迹--------------#
                    #ax = plt.subplot(subplt_grid[0, 0:2], projection = '3d')
                    ax = fig_all.add_subplot(111, projection='3d')
                    #ax.set_title('Auditory Navigation Trajectory')
                    space_x, space_y, space_z = space_size
                    for s in [0, 1]:
                        for t in [0, 1]:
                            ax.plot([0, space_x], [s*space_y, s*space_y], [t*space_z, t*space_z], 'k-', alpha=0.3)
                            ax.plot([s*space_x, s*space_x], [0, space_y], [t*space_x, t*space_z], 'k-', alpha=0.3)
                            ax.plot([s*space_x, s*space_x], [s*space_y, s*space_y], [0, space_z], 'k-', alpha=0.3)
                    """
                    # 绘制障碍物（正方体）
                    obs_center, side_length = env.obstacle
                    half_side = side_length / 2
    
                    # 定义正方体的8个顶点
                    min_bounds = np.array(obs_center) - half_side
                    max_bounds = np.array(obs_center) + half_side
    
                    vertices = np.array([
                        [min_bounds[0], min_bounds[1], min_bounds[2]],
                        [max_bounds[0], min_bounds[1], min_bounds[2]],
                        [max_bounds[0], max_bounds[1], min_bounds[2]],
                        [min_bounds[0], max_bounds[1], min_bounds[2]],
                        [min_bounds[0], min_bounds[1], max_bounds[2]],
                        [max_bounds[0], min_bounds[1], max_bounds[2]],
                        [max_bounds[0], max_bounds[1], max_bounds[2]],
                        [min_bounds[0], max_bounds[1], max_bounds[2]],
                    ])
    
                    # 定义正方体的6个面
                    faces = [
                        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
                        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
                        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
                        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
                    ]
    
                    # 创建正方体障碍物对象并添加到图中
                    cube = Poly3DCollection(faces, alpha=0.8, edgecolor='k', linewidths=1)
                    cube.set_facecolor('gold')
                    ax.add_collection3d(cube)
                    """
                    
                    #Agent移动路径
                    pos = np.array(Agent_pos_history)
                    ax.plot(pos[:,0], pos[:,1], pos[:,2], color='#1E90FF', linestyle='-', label='Path')
                    #设置字体为新罗马
                    plt.rcParams["font.family"] = "Times New Roman"
                    # 绘制Agent初始位置点并添加标注
                    agent_start = ax.scatter(pos[0,0], pos[0,1], pos[0,2], c='blue', s=150, depthshade=False)
                    ax.text(pos[0,0], pos[0,1], pos[0,2]+0.5, "Start", 
                            fontsize=19, color='black', 
                            horizontalalignment='center',   # 水平居中对齐
                            verticalalignment='bottom',     # 垂直底部对齐（点上方）
                            zorder=5)                       # 设置较高的zorder确保文本显示在顶部
                    # 绘制Agent结束位置点并添加标注
                    agent_end = ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], c='navy', s=150, depthshade=False)
                    ax.text(pos[-1,0], pos[-1,1], pos[-1,2]+0.5, "End", 
                            fontsize=19, color='black', 
                            horizontalalignment='center',   # 水平居中对齐
                            verticalalignment='bottom',     # 垂直底部对齐（点上方）
                            zorder=5)                       # 设置较高的zorder确保文本显示在顶部
                    #声源位置
                    sound_source = ax.scatter(source_target[0], source_target[1], source_target[2], c='#00FF00', s=1700)
                    ax.text(source_target[0], source_target[1], source_target[2]+0.5, "Sound Source", 
                            fontsize=19, color='black', 
                            horizontalalignment='center',   # 水平居中对齐
                            verticalalignment='bottom',     # 垂直底部对齐（点上方）
                            zorder=10)                      # 设置较高的zorder确保文本显示在顶部
                    #声源距离测量误差
                    #if_obstructed = env._is_obstructed(source_target, pos[-1])
                    dis_error = abs(esti_dist_history[-1] - true_dist_history[-1])  
                    rel_error = (dis_error / true_dist_history[-1]) * 100
                    ax.text2D(0.05, 0.95, 
                            #f"Obstructed: {if_obstructed}\n"
                            f"True Dist: {true_dist_history[-1]:.2f}m\n"
                            f"Est Dist: {esti_dist_history[-1]:.2f}m\n"
                            f"Error: {dis_error:.2f}m ({rel_error:.1f}%)",
                            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
                    ax.set_xlabel('x(m)')
                    ax.set_ylabel('y(m)')
                    ax.set_zlabel('z(m)')
                    ax.set_title('Auditory Navigation Trajectory')
                    #设置坐标轴范围与房间尺寸一致
                    ax.set_xlim(0, space_x)
                    ax.set_ylim(0, space_y)
                    ax.set_zlim(0, space_z)
    
                    # 添加图例
                    from matplotlib.patches import Rectangle
                    proxy = [#Rectangle((0,0),1,1,fc='gold', alpha=0.8),
                            plt.Line2D([0], [0], color='#1E90FF', lw=2),
                            plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.5),
                            plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor='blue'),
                            plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor='navy'),
                            plt.Line2D([0], [0], marker='o', color='w', markersize=15, markerfacecolor='#00FF00')]
                    ax.legend(proxy, ['Agent Path', 'Signal Path', 'Start', 'End', 'Sound Source'],
                            loc='lower left', fontsize=12)
                    plt.tight_layout()
                        #plt.show()
                 
                    plt.gcf().savefig(f'./{path}{path2}{R+1}-th_times_{Tr+1}-th_trials_{int(n/5)}-steps.png',transparent = False, facecolor = 'white', dpi=200)
                    plt.close()
                    
                    n += 1
                
                if arr_flag == 1:
                    break
            decay = decay * 0.98
                
    end_time = perf_counter()
    print('运行总时间:', end_time - start_time)
    data['args'] = args
    data['sound_source_position'] = [source_target[0], source_target[1], source_target[2]]             
    data['weights_update_times'] = weights_update_times
    
    savemat(f'./{path}-data{current_datetime}.mat', data,  do_compression=True, appendmat=False, format='5', long_field_names=False, oned_as='row')
    #保存为json格式文件
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将 NumPy 数组转换为列表
        elif isinstance(obj, np.generic):
            return obj.item()    # 将 NumPy 标量转换为 Python 标量
        elif isinstance(obj, datetime):
            return obj.isoformat()  # 将日期时间转换为字符串
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
                  
    json_file_path = f"./{path}-data{current_datetime}.json"  
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(
            data, 
            f,
            ensure_ascii=False,  # 保持非 ASCII 字符（如中文）
            indent=2,            # 缩进 2 个空格（对应 MAT 文件的结构化显示）
            default=convert_to_json_serializable,  # 处理特殊类型
            sort_keys=False      # 不排序键（保持原始顺序）
        )

    print(f"数据已保存为 JSON 文件: {json_file_path}")              
                        
if __name__=="__main__":
    main()                   
                  
            
            
    