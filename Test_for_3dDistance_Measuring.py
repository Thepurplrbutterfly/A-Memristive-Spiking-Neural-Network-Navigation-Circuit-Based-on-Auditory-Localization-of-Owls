import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import distance
from scipy.signal import chirp, correlate

class AcousticEnvironment:
    def __init__(self, room_size=(50, 50, 50), obstacle=None):
        """
        初始化三维声学环境
        :param room_size: (长, 宽, 高) 单位米
        :param obstacle: ((x,y,z), 半径)
        """
        self.room_size = room_size
        self.obstacle = obstacle if obstacle else ((25, 25, 25), 5)  # 中心球体障碍物
        self.sound_speed = 343  # 声速 (m/s)
        self.sampling_rate = 192000  # 采样率 (Hz)

    def generate_sound(self, duration=0.15, freq_range=(500, 6000)):
        """生成带汉宁窗的线性调频信号"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration), False)
        signal = chirp(t, f0=freq_range[0], f1=freq_range[1], t1=duration)
        window = np.hanning(len(signal))
        return (signal * window) / np.max(np.abs(signal))

    def _is_obstructed(self, source_pos, receiver_pos):
        """三维线段与球体的碰撞检测"""
        obs_pos, obs_radius = self.obstacle
        line_vec = np.array(receiver_pos) - np.array(source_pos)
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return False
            
        # 计算点到线段的最小距离
        obs_to_source = np.array(source_pos) - np.array(obs_pos)
        dot_product = np.dot(obs_to_source, line_vec)
        t = -dot_product / (line_len**2)
        t = np.clip(t, 0, 1)  # 限制在线段范围内
        nearest_point = source_pos + t * line_vec
        
        # 球体距离判断
        dist_to_center = np.linalg.norm(nearest_point - obs_pos)
        return dist_to_center < obs_radius

    def propagate_sound(self, source_pos, receiver_pos, signal):
        """三维声传播模型（含绕射和衰减）"""
        true_dist = np.linalg.norm(np.array(source_pos) - np.array(receiver_pos))
        
        # 计算有效传播距离
        if self._is_obstructed(source_pos, receiver_pos):
            obs_pos, obs_radius = self.obstacle
            vec1 = np.array(source_pos) - np.array(obs_pos)
            vec2 = np.array(receiver_pos) - np.array(obs_pos)
            
            # 三维空间中的夹角计算
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            theta = np.arccos(np.clip(cos_theta, -1, 1))
            detour_dist = (np.linalg.norm(vec1) + np.linalg.norm(vec2) + obs_radius * theta) * 1.1
            attenuation = 0.15 / (true_dist ** 1.5)  # 三维障碍衰减系数
        else:
            detour_dist = true_dist
            attenuation = 1.0 / (true_dist ** 1.1)  # 三维自由场衰减
            
        # 计算延迟样本数
        delay_samples = int((detour_dist / self.sound_speed) * self.sampling_rate)
        
        # 接收信号缓冲区
        buffer_length = len(signal)*8 + delay_samples
        received_signal = np.zeros(buffer_length)
        
        # 添加距离自适应的噪声
        noise_level = 0.005 * (1 + true_dist/50)
        noise = np.random.normal(0, noise_level, buffer_length)
        received_signal[delay_samples:delay_samples+len(signal)] = signal * attenuation
        received_signal += noise
        
        return received_signal, true_dist


class AcousticPerceptionSystem:
    def __init__(self, initial_pos, environment):
        """
        三维听觉感知系统
        :param initial_pos: (x, y, z) 初始位置
        :param environment: 声学环境对象
        """
        self.position = np.array(initial_pos)
        self.environment = environment
        self.estimated_distances = []
        self.true_distances = []
        self.history_positions = [self.position.copy()]

    def move_randomly(self, step_size=4.0):
        """三维随机移动"""
        # 生成随机方向向量（球面均匀分布）
        theta = np.random.uniform(0, 2*np.pi)  # 水平角
        phi = np.arccos(2*np.random.uniform(0, 1) - 1)  # 俯仰角
        direction = np.array([
            np.sin(phi)*np.cos(theta),
            np.sin(phi)*np.sin(theta),
            np.cos(phi)
        ])
        
        new_pos = self.position + step_size * direction
        
        # 三维边界约束
        new_pos = np.clip(new_pos, 0, self.environment.room_size)
        self.position = new_pos
        self.history_positions.append(self.position.copy())

    def estimate_distance(self, source_signal):
        """改进的三维距离估计"""
        # 生成声源位置（远离边界）
        margin = min(5, self.environment.room_size[0]/10)
        source_pos = (
            np.random.uniform(margin, self.environment.room_size[0]-margin),
            np.random.uniform(margin, self.environment.room_size[1]-margin),
            np.random.uniform(margin, self.environment.room_size[2]-margin)
        )
        
        # 获取接收信号
        received_signal, true_dist = self.environment.propagate_sound(
            source_pos, self.position, source_signal)
        
        # 互相关计算（FFT加速）
        corr = correlate(received_signal, source_signal, mode='full', method='fft')
        
        # 能量检测确定有效信号段
        energy = np.cumsum(np.abs(received_signal))
        threshold = 0.1 * energy[-1]
        signal_start = np.argmax(energy > threshold)
        signal_end = len(energy) - np.argmax(energy[::-1] > threshold)
        
        # 主峰检测
        search_start = max(0, signal_start - len(source_signal)//2)
        search_end = min(len(corr), signal_end + len(source_signal)//2)
        corr_segment = corr[search_start:search_end]
        main_peak = np.argmax(np.abs(corr_segment)) + search_start
        
        # 二次插值精确定位
        if 1 < main_peak < len(corr)-1:
            delta = 0.5*(corr[main_peak-1]-corr[main_peak+1]) / \
                   (corr[main_peak-1]-2*corr[main_peak]+corr[main_peak+1])
            main_peak += delta
        
        # 计算距离
        delay_samples = main_peak - (len(source_signal) - 1)
        time_delay = abs(delay_samples) / self.environment.sampling_rate
        estimated_dist = time_delay * self.environment.sound_speed
        
        # 物理约束
        max_dist = np.sqrt(sum(d**2 for d in self.environment.room_size))
        estimated_dist = np.clip(estimated_dist, 0.5, max_dist*1.1)
        
        # 保存结果
        self.estimated_distances.append(estimated_dist)
        self.true_distances.append(true_dist)
        
        return estimated_dist, true_dist, source_pos


def visualize(environment, perception_system, source_pos):
    """三维可视化"""
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    #ax.grid(False)
    # 绘制房间框架
    room_x, room_y, room_z = environment.room_size
    for s in [0, 1]:
        for t in [0, 1]:
            ax.plot([0, room_x], [s*room_y, s*room_y], [t*room_z, t*room_z], 'k-', alpha=0.3)
            ax.plot([s*room_x, s*room_x], [0, room_y], [t*room_z, t*room_z], 'k-', alpha=0.3)
            ax.plot([s*room_x, s*room_x], [t*room_y, t*room_y], [0, room_z], 'k-', alpha=0.3)
    
    # 绘制障碍物（球体）
    obs_pos, obs_radius = environment.obstacle
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = obs_pos[0] + obs_radius * np.outer(np.cos(u), np.sin(v))
    y = obs_pos[1] + obs_radius * np.outer(np.sin(u), np.sin(v))
    z = obs_pos[2] + obs_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='gold', alpha=0.8)
    ax.legend(fontsize=12)
    # 绘制移动路径
    path = np.array(perception_system.history_positions)
    ax.plot(path[:,0], path[:,1], path[:,2], color='#1E90FF', linestyle='-')
    # 设置字体为新罗马
    plt.rcParams["font.family"] = "Times New Roman"

    # 绘制Agent初始位置点并添加标注（位于点上方）
    agent_start = ax.scatter(path[0,0], path[0,1], path[0,2], c='blue', s=150, depthshade=False)
    ax.text(path[0,0], path[0,1], path[0,2], "Start", 
            fontsize=19, color='black', 
            horizontalalignment='center',   # 水平居中对齐
            verticalalignment='bottom',     # 垂直底部对齐（点上方）
            zorder=5)                       # 设置较高的zorder确保文本显示在顶部

    # 绘制Agent结束位置点并添加标注（位于点上方）
    agent_end = ax.scatter(path[-1,0], path[-1,1], path[-1,2], c='navy', s=150, depthshade=False)
    ax.text(path[-1,0], path[-1,1], path[-1,2], "End", 
            fontsize=19, color='black', 
            horizontalalignment='center',   # 水平居中对齐
            verticalalignment='bottom',     # 垂直底部对齐（点上方）
            zorder=5)                       # 设置较高的zorder确保文本显示在顶部

    # 绘制声源位置并添加标注（位于点上方）
    sound_source = ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='#00FF00', s=1700)
    ax.text(source_pos[0], source_pos[1], source_pos[2], "Sound Source", 
            fontsize=19, color='black', 
            horizontalalignment='center',   # 水平居中对齐
            verticalalignment='bottom',     # 垂直底部对齐（点上方）
            #bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'), # 可选：添加半透明背景框
            zorder=10)                       # 设置较高的zorder确保文本显示在顶部

    """
    ax.text(
    source_pos[0], source_pos[1], source_pos[2],
    'Sound Source',
    color='black',
    fontsize=12,
    ha='center', va='center',
    zorder=2  # 文字层级（高于圆形）
    )
    """
    #circle = plt.Circle((source_pos[0], source_pos[1]), 3, color='#00FF00', fill=True)
    #plt.gcf().gca().add_artist(circle)
    ax.plot([path[-1,0], source_pos[0]], 
            [path[-1,1], source_pos[1]], 
            [path[-1,2], source_pos[2]], 'r--', alpha=0.3)
    
    # 添加信息标注
    is_obstructed = environment._is_obstructed(source_pos, path[-1])
    error = abs(perception_system.estimated_distances[-1] - perception_system.true_distances[-1])
    rel_error = error / perception_system.true_distances[-1] * 100
    
    ax.text2D(0.05, 0.95, 
             f"Obstructed: {is_obstructed}\n"
             f"True Dist: {perception_system.true_distances[-1]:.2f}m\n"
             f"Est Dist: {perception_system.estimated_distances[-1]:.2f}m\n"
             f"Error: {error:.2f}m ({rel_error:.1f}%)",
             transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Acoustic Perception System')
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    # 初始化三维环境
    env = AcousticEnvironment(room_size=(50, 50, 50))
    
    # 创建感知系统（初始位置设为(5,5,5)）
    perception_system = AcousticPerceptionSystem(initial_pos=(5, 5, 5), environment=env)
    
    # 生成声源信号
    source_signal = env.generate_sound()
    
    # 模拟运行
    for i in range(15):
        # 随机移动
        perception_system.move_randomly(step_size=4.0)
        
        # 距离估计
        est_dist, true_dist, source_pos = perception_system.estimate_distance(source_signal)
        
        print(f"Step {i+1}: Position: {perception_system.position.round(2)} | "
              f"Source: {np.array(source_pos).round(2)} | "
              f"Distance: {est_dist:.2f}m (True: {true_dist:.2f}m) | "
              f"Error: {abs(est_dist-true_dist):.2f}m")
        
        # 可视化
        visualize(env, perception_system, source_pos)
        time.sleep(0.5)  # 控制演示速度


if __name__ == "__main__":
    main()