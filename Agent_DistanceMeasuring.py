import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from math import cos
from math import sin
from scipy.spatial import distance   #distance为scipy中用于计算空间距离的模块
from scipy.signal import chirp, correlate   #chirp用于生成线性和频率随时间变化的非线性调频信号
                                            #correlate用于计算两个信号的互相关，时延估计
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
class Auditory_Environment:
    def __init__(self, space_size, obstacle=None):  
        """
        构建一个听觉三维空间设置类：
        将空间大小默认设置为50m*50m*50,且障碍物默认为None,障碍物若存在可后续调用时自行设置位置和尺寸
        """
        self.space_size = space_size
        self.obstacle = obstacle if obstacle else ((25, 25, 25), 8)  #自行设置障碍物坐标(25, 25, 25)和半径5m
        """
        obstacle if obstacle else ((25, 25, 25), 5)等价于
        if obstacle:
           self.obstacle = obstacle  #传入的参数
        else:
           self.obstacle = ((25, 25, 25), 5)
        """
        self.sound_speed = 340 #空气中声速设置(m/s)
        self.sampling_rate = 192000  #声源采样频率，设置对产生的连续模拟声波信号每秒采集的样本数
           
    def sound_source(self, duration=0.15, freq_range=(500, 6000)):  
        """
        声源声波信号产生函数：
        duration:产生声波的持续时间(s),采样点=duration*sampling_rate,所以如果要修改duration,则要对sampling_rate进行相应的更改
        freq_range:产生的chirp声波信号的频率变化范围,人类听觉范围约为 20Hz ~ 20000Hz,而语音主要集中在 300Hz ~ 3400Hz
        """  
        t = np.linspace(0, duration, int(self.sampling_rate * duration), False)  #生成一个时间轴数组，存储所有采样时间点
        ss_signal = chirp(t, f0=freq_range[0], f1=freq_range[1], t1=duration)  #生成指定初始频率和截止频率的chirp非线性调频信号模拟语音
        window = np.hanning(len(ss_signal))  #减少信号截断处的突变，降低频谱泄露，提高TOA检测精度
        return (ss_signal * window) / np.max(np.max(ss_signal))  #对采样得到的信号进行归一化
    
    def sound_propagate(self, source_pos, Agent_pos, ss_signal):
        """
        模拟声音传播，考虑障碍物遮挡和距离衰减
        返回接收到的信号和真实几何距离
        """
        #声源点和Agent点的直线距离
        Euclidean_dist = distance.euclidean(source_pos, Agent_pos)  
        #计算声音的有效传播距离:
        """
        判断source_pos和Agent_pos之间的直线距离是否被障碍物遮挡
        """
        if self._is_obstructed(source_pos, Agent_pos):  #当被障碍物遮挡
            #obstacle_pos, obstacle_radius = self.obstacle
            #计算障碍物位于直线路径时的实际路径(绕射路径)
            Detour_dist = Euclidean_dist * 1.35  # 正方体绕射系数
            #Detour_dist为遇障碍物后的实际传播距离，这里定义为直线距离+偏转角对应的弧度
            attenuation = 0.18 / (Euclidean_dist ** 1.6)  #设置障碍物对声音的衰减系数
        else:  #未被障碍物遮挡
            Detour_dist = Euclidean_dist
            attenuation = 1 / (Euclidean_dist ** 0.9)      
        #延迟过程中声源产生的声音样本数，T_delay_samples为真实产生的时延理论值
        T_delay_samples = int((Detour_dist / self.sound_speed) * self.sampling_rate)
        #扩展Agent接收信号的缓冲区->放置内存外泄
        buffer_length = len(ss_signal) * 8 + T_delay_samples   #缓冲区大小为声源产生信号长度的8倍+延迟时间内的样点数
        Agent_Resignal = np.zeros(buffer_length)
        #添加距离自适应噪声来模拟环境中的噪声
        noise_intensity = 0.005 * (1 + Euclidean_dist / 50) #噪声强度随距离的增加而增大
        noise = np.random.normal(0, noise_intensity, buffer_length)  #生成均值为0，标准差为noise_intensith，长度为buffer_length的服从高斯分布的噪声数组
        Agent_Resignal[T_delay_samples:T_delay_samples+len(ss_signal)] = ss_signal * attenuation
        Agent_Resignal += noise 
        return Agent_Resignal, Euclidean_dist
    
    def _is_obstructed(self, source_pos, Agent_pos):
        """声音传播路径与正方体障碍的碰撞检测"""
        if self.obstacle == None:
            return False
        
        obs_center, side_length = self.obstacle
        half_side = side_length / 2
        
        #正方体坐下角和右上角坐标计算
        min_bounds = np.array(obs_center) - half_side
        max_bounds = np.array(obs_center) + half_side
        
        #射线与正方体的碰撞检测slab算法
        source_pos = np.array(source_pos)
        Agent_pos = np.array(Agent_pos)
        vector_as = Agent_pos - source_pos
        M_vector_as = np.linalg.norm(vector_as)
        
        if M_vector_as == 0:
            return False
        
        #标准化方向向量
        vector_as = vector_as / M_vector_as
        
        tmin = 0.0
        tmax = M_vector_as
        
        #在三个坐标轴上的碰撞检测
        for i in range(3):
            if abs(vector_as[i]) < 1e-6: #此时射线与对应平面平行
                if source_pos[i] < min_bounds[i] or source_pos[i] > max_bounds[i]:
                    """此时射线必不会与障碍物相交"""
                    return False
            else:
                t1 = (min_bounds[i] - source_pos[i]) / vector_as[i]
                t2 = (max_bounds[i] - source_pos[i]) / vector_as[i]
                #预设t1为进入障碍物的射线,t2为离开障碍物的射线,但由于声源位置未知,所以二者互换
                if t1 > t2:  #t2实际上为进入障碍物的射线
                    t1, t2 = t2, t1
                """每次遍历都会对tmin进行更新,从而找出三个平面最后射入障碍物的射线"""    
                if t1 > tmin:
                    tmin = t1
                """每次遍历都会对tmax进行更新,从而找出三个平面最先射出障碍物的射线"""     
                if t2 < tmax:
                    tmax = t2
                    
                if tmin > tmax:
                    return False
                
        return tmin >= 0 and tmax >= 0 and tmin <= M_vector_as and tmax <= M_vector_as
                
class Auditory_Perception:
    def __init__(self, initial_pos, env):
        """
        构建一个听觉感知类
        initial_pos:Agent初始位置
        env:构建的听觉三维空间类
        """
        self.Agent_pos = np.array(initial_pos)
        self.env = env
        self.est_dist = []  #初始化一个用于存储系统估计距离的空列表，列表大小动态可变
        self.real_dist = []  #初始化一个用于存储实际距离的空列表，列表大小动态可变
        self.history_positions = [self.Agent_pos.copy()]  #初始化一个用于存储Agent历史轨迹的空列表，列表大小动态可变，且列表初始值为Agent当前自身位置的副本self.pos.copy()
    
    def move_directly(self, ec_Angle_Array, step=5.0):  
        #ec_Angle_Array为群体编码后水平角和垂直角的数组，每个元素以弧度表示
        encoded_HAngle = ec_Angle_Array[0]
        encoded_VAngle = ec_Angle_Array[1]
        theta_h = encoded_HAngle
        theta_v = encoded_VAngle
        if self.est_dist > 3:
            self.Agent_pos[0] += (step * cos(theta_v)) * sin(theta_h)
            self.Agent_pos[1] += (step * cos(theta_v)) * cos(theta_h)
            self.Agent_pos[2] += step * sin(theta_v)
        self.history_positions.append(self.Agent_pos.copy())
        return self.Agent_pos
    
    def dist_estimating(self, ss_signal, source_target):
        #声源位置生成
        source_pos = source_target  #固定生成声源位置[32, 32, 32]
        """
        #随机生成声源位置：
        source_pos = (np.random.uniform(25, self.env.space_size[0]), np.random.uniform(25, self.env.space_size[1]), np.random.uniform(25, self.env.space_size[2]))
        #np.random.uniform(low, high]   生成[low, high)内的随及浮点数
        """
        #Agent获取声源产生的声音信号
        Agent_Resignal, Euclidean_dist = self.env.sound_propagate(source_pos, self.Agent_pos, ss_signal)   
        #互相关计算
        corr = correlate(Agent_Resignal, ss_signal, mode='full', method='fft')
        """
        互相关函数用于衡量两个信号在时间偏移存在时的相似性：
        这里用于比较声源信号ss_signal和Agent接收到的信号Agent_Resignal之间的相似性
        若ss_signal=S[n], Agent_Resignal=S[n+Δt]+noise, noise为干扰, t为偏移的时间, 即传播所耗时间
        那么互相关函数R(t)会在Δt处出现峰值
        """
        #通过累加信号并设定阈值来确定有效信号段
        Agent_Resignal_sum = np.cumsum(np.abs(Agent_Resignal))  #Agent_Resignal_sum仍是与Agent_Resignal同size的数组
        threshold = 0.1 * Agent_Resignal_sum[-1]   #将判断阈值设为信号采样点累加和的10%，Agent_Resignal_sum[-1]为数组中最后一个值
        start_signal = np.argmax(Agent_Resignal_sum > threshold)   #返回累加信号中首次大于threshold的索引作为有效段起始点
        end_signal = len(Agent_Resignal_sum) - np.argmax(Agent_Resignal_sum[::-1] > threshold)   ##返回累加信号中最后大于threshold的索引作为有效段结束点
        """Agent_Resignal_sum[::-1]是将数组中的元素的位置进行翻转, 以通过np.argmax()获取对应索引"""
        """
        因为信号中可能存在噪声或者无效段, 因此最好通过阈值判定来决定信号检测的有效段
        """
        #互相关计算结果中的峰值检测
        search_startp = max(0, start_signal - len(ss_signal) // 2)  #检测起始点
        search_stopp = min(len(corr), end_signal + len(ss_signal // 2))  #检测结束点
        search_segment = corr[search_startp:search_stopp]
        main_peak_point = np.argmax(np.abs(search_segment)) + search_startp
        """检测起始点和结束点较有效段起始点和结束点偏移N/2是为了捕捉完整的互相关峰"""
        #二值插值提高检测精度
        if 1 < main_peak_point < len(corr)-1:
            delta = 0.5 * (corr[main_peak_point - 1] - corr[main_peak_point + 1]) / \
                    (corr[main_peak_point - 1] - 2 * corr[main_peak_point] + corr[main_peak_point + 1])
            main_peak_point += delta
        #互相关法计算声源距离(估计的距离)
        E_delay_samples = main_peak_point - (len(ss_signal) - 1)
        """
        互相关函数的结果长度为N+M-1, 其中N为声源信号的长度, M为Agent接收信号的长度, 当无时延时
        峰值位于N-1处, 即零时偏移, 通过main_peak_point-(N-1)就获得实际偏移量
        """
        E_time_delay = E_delay_samples / self.env.sampling_rate
        estimated_dist = E_time_delay * self.env.sound_speed
        #对估计距离进行约束
        max_dist = np.sqrt(sum(d**2 for d in self.env.space_size))
        np.clip(estimated_dist, 0.5, max_dist * 1.1)
        
        #结果保存
        self.est_dist.append(estimated_dist)
        self.real_dist.append(Euclidean_dist)
        return estimated_dist, Euclidean_dist, source_pos
    
#可视化函数
def visualize_3D(env, Ag):
    """三维可视化"""
    fig = plt.figure(fig_size=(14, 14))
    ax = fig.add_subplot(111, projection='3d')  #创建一个三维子图
    """
    ax为创建的三维框架, fig为建立的图形窗口, add_subplot()用于在fig中的指定位置放置子图(坐标轴对象)
    projection="3d"用于将子图设置为三维投影模式
    """
    #绘制三维空间框架
    space_x, space_y, space_z = env.space_size
    """绘制12条正方体空间框架边界线"""
    for i in [0, 1]:
        for j in [0, 1]:
            ax.plot([0, space_x], [i * space_y, i * space_y], [j * space_z, j * space_z], 'k-', alpha=0.3)  #绘制四条x轴上的边界框线， k-用于将线设置为黑色实线，0.3为线条的透明度
            ax.plot([i * space_x, i * space_x], [0, space_y], [j * space_z, j * space_z], 'k-', alpha=0.3)  #绘制四条y轴上的边界框线， k-用于将线设置为黑色实线，0.3为线条的透明度
            ax.plot([i * space_x, i * space_x], [j * space_y, j * space_y], [0, space_z], 'k-', alpha=0.3)  #绘制四条z轴上的边界框线， k-用于将线设置为黑色实线，0.3为线条的透明度
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
    
    # 创建正方体对象并添加到图中
    cube = Poly3DCollection(faces, alpha=0.8, edgecolor='k', linewidths=1)
    cube.set_facecolor('gold')
    ax.add_collection3d(cube)
    #绘制Agent移动路径
    path = np.array(Ag.history_positions)
    ax.plot(path[:,0], path[:,1], path[:,2], color='#1E90FF', linestyle='-', label='Agent Movement Path')
    # 设置字体为新罗马
    plt.rcParams["font.family"] = "Times New Roman"
    # 绘制Agent初始位置点并添加标注
    agent_start = ax.scatter(path[0,0], path[0,1], path[0,2], c='blue', s=150, depthshade=False)
    ax.text(path[0,0], path[0,1], path[0,2]+0.5, "Start", 
            fontsize=19, color='black', 
            horizontalalignment='center',   # 水平居中对齐
            verticalalignment='bottom',     # 垂直底部对齐（点上方）
            zorder=5)                       # 设置较高的zorder确保文本显示在顶部
    # 绘制Agent结束位置点并添加标注
    agent_end = ax.scatter(path[-1,0], path[-1,1], path[-1,2], c='navy', s=150, depthshade=False)
    ax.text(path[-1,0], path[-1,1], path[-1,2]+0.5, "End", 
            fontsize=19, color='black', 
            horizontalalignment='center',   # 水平居中对齐
            verticalalignment='bottom',     # 垂直底部对齐（点上方）
            zorder=5)                       # 设置较高的zorder确保文本显示在顶部
    #绘制声源散点
    sound_source = ax.scatter(Ag.source_pos[0], Ag.source_pos[1], Ag.source_pos[2], c='#00FF00', s=1700)
    ax.text(Ag.source_pos[0], Ag.source_pos[1], Ag.source_pos[2]+0.5, "Sound Source", 
            fontsize=19, color='black', 
            horizontalalignment='center',   # 水平居中对齐
            verticalalignment='bottom',     # 垂直底部对齐（点上方）
            zorder=10)                      # 设置较高的zorder确保文本显示在顶部
    error = abs(Ag.est_dist[-1] - Ag.real_dist[-1])  #测量声距与实际理论声距间的差值
    err_percentage = (error / Ag.real_dist[-1]) * 100  #误差百分比
    
    print(f"Euclidean distance:{Ag.real_dist[-1]}m", end="")
    print(f"Estimated distance:{Ag.est_dist[-1]}m", end="")
    print(f"Error:{error}m", end="")
    print(f"Error Percentage:{err_percentage}%", end="")
    
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')
    ax.set_title('Auditory-based Agent Navigation Path')
    # 设置坐标轴范围与房间尺寸一致
    ax.set_xlim(0, space_x)
    ax.set_ylim(0, space_y)
    ax.set_zlim(0, space_z)
    
    # 添加图例
    from matplotlib.patches import Rectangle
    proxy = [Rectangle((0,0),1,1,fc='gold', alpha=0.8),
             plt.Line2D([0], [0], color='#1E90FF', lw=2),
             plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.5),
             plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor='blue'),
             plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor='navy'),
             plt.Line2D([0], [0], marker='o', color='w', markersize=15, markerfacecolor='#00FF00')]
    ax.legend(proxy, ['Cube Obstacle', 'Agent Path', 'Signal Path', 'Start', 'End', 'Sound Source'],
              loc='lower left', fontsize=12)
    #ax.legend()  #图里添加函数
    plt.tight_layout()  #自动调整布局
    plt.show()  #显示图窗 
        
          
        
        
        