import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
from scipy.signal import chirp, correlate

class AcousticEnvironment:
    def __init__(self, room_size=(50, 50), obstacle=None):  # 修改为50x50
        """
        初始化声学环境
        :param room_size: 房间尺寸 (长, 宽) 默认50x50米
        :param obstacle: 障碍物位置和尺寸 ((x, y), 半径)
        """
        self.room_size = room_size
        self.obstacle = obstacle if obstacle else ((25, 25), 2.5)  # 障碍物保持在中心但尺寸不变
        self.sound_speed = 343  # 空气中声速 (m/s)
        self.sampling_rate = 192000  # 进一步提高采样率以适应更大空间
        
    def generate_sound(self, duration=0.15, freq_range=(500, 6000)):  # 延长持续时间并调整频率范围
        """生成带汉宁窗的线性调频信号"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration), False)
        signal = chirp(t, f0=freq_range[0], f1=freq_range[1], t1=duration)
        window = np.hanning(len(signal))
        return (signal * window) / np.max(np.abs(signal))
    
    def propagate_sound(self, source_pos, receiver_pos, signal):
        """
        模拟声音传播，考虑障碍物遮挡和距离衰减
        返回接收到的信号和真实几何距离
        """
        true_dist = distance.euclidean(source_pos, receiver_pos)
        
        # 计算有效传播距离（考虑绕射）
        if self._is_obstructed(source_pos, receiver_pos):
            obs_pos, obs_radius = self.obstacle
            # 精确绕射路径计算（基于几何光学）
            vec1 = np.array(source_pos) - np.array(obs_pos)
            vec2 = np.array(receiver_pos) - np.array(obs_pos)
            theta = np.arccos(np.clip(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)), -1, 1))
            detour_dist = (np.linalg.norm(vec1) + np.linalg.norm(vec2) + obs_radius * theta) * 1.1  # 增加绕射余量
            attenuation = 0.2 / (true_dist ** 1.3)  # 调整障碍物衰减系数
        else:
            detour_dist = true_dist
            attenuation = 1.0 / (true_dist ** 0.9)  # 调整自由场衰减系数
        
        # 计算延迟样本数（去除强制最小值限制）
        delay_samples = int((detour_dist / self.sound_speed) * self.sampling_rate)
        
        # 扩展接收信号缓冲区（8倍信号长度 + 延迟）
        buffer_length = len(signal)*8 + delay_samples
        received_signal = np.zeros(buffer_length)
        
        # 添加距离自适应的噪声
        noise_level = 0.005 * (1 + true_dist/50)  # 噪声随距离增加
        noise = np.random.normal(0, noise_level, buffer_length)
        received_signal[delay_samples:delay_samples+len(signal)] = signal * attenuation
        received_signal += noise
        
        return received_signal, true_dist
    
    def _is_obstructed(self, source_pos, receiver_pos):
        """检查声源和接收器之间是否有障碍物阻挡"""
        obs_pos, obs_radius = self.obstacle
        x1, y1 = source_pos
        x2, y2 = receiver_pos
        x0, y0 = obs_pos
        
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + (x2-x1)*y1 - (y2-y1)*x1)
        denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        if denominator == 0:
            return False
        
        dist_to_line = numerator / denominator
        if dist_to_line < obs_radius:
            dot_product = (x0 - x1)*(x2 - x1) + (y0 - y1)*(y2 - y1)
            if 0 < dot_product < (x2 - x1)**2 + (y2 - y1)**2:
                return True
        return False


class AcousticPerceptionSystem:
    def __init__(self, initial_pos, environment):
        """
        初始化听觉感知系统
        :param initial_pos: 初始位置 (x, y)
        :param environment: 声学环境对象
        """
        self.position = np.array(initial_pos)
        self.environment = environment
        self.estimated_distances = []
        self.true_distances = []
        self.history_positions = [self.position.copy()]
        
    def move_randomly(self, step_size=2.0):  # 增大步长适应更大空间
        """随机移动感知系统"""
        angle = np.random.uniform(0, 2*np.pi)
        new_pos = self.position + step_size * np.array([np.cos(angle), np.sin(angle)])
        
        # 确保不会移动到房间外
        new_pos[0] = np.clip(new_pos[0], 0, self.environment.room_size[0])
        new_pos[1] = np.clip(new_pos[1], 0, self.environment.room_size[1])
        
        self.position = new_pos
        self.history_positions.append(self.position.copy())
        
    def estimate_distance(self, source_signal):
        """
        通过接收到的信号估计与声源的距离（改进版）
        """
        # 生成声源位置（确保不靠近边界）
        margin = min(5, self.environment.room_size[0]/10)  # 动态边界余量
        source_pos = (
            np.random.uniform(margin, self.environment.room_size[0]-margin),
            np.random.uniform(margin, self.environment.room_size[1]-margin)
        )
        
        # 获取接收信号和真实距离
        received_signal, true_dist = self.environment.propagate_sound(
            source_pos, self.position, source_signal)
        
        # ----------------- 改进的TOA估计 -----------------
        # 使用FFT加速的互相关计算
        corr = correlate(received_signal, source_signal, mode='full', method='fft')
        
        # 能量检测确定有效信号段
        energy = np.cumsum(np.abs(received_signal))
        threshold = 0.1 * energy[-1]
        signal_start = np.argmax(energy > threshold)
        signal_end = len(energy) - np.argmax(energy[::-1] > threshold)
        
        # 仅在有效段寻找主峰
        search_start = max(0, signal_start - len(source_signal)//2)
        search_end = min(len(corr), signal_end + len(source_signal)//2)
        corr_segment = corr[search_start:search_end]
        
        main_peak = np.argmax(np.abs(corr_segment)) + search_start
        
        # 二次插值精确定位峰值
        if 1 < main_peak < len(corr)-1:
            delta = 0.5*(corr[main_peak-1]-corr[main_peak+1]) / \
                   (corr[main_peak-1]-2*corr[main_peak]+corr[main_peak+1])
            main_peak += delta
        
        # 计算实际延迟（修正公式）
        delay_samples = main_peak - (len(source_signal) - 1)
        time_delay = abs(delay_samples) / self.environment.sampling_rate
        estimated_dist = time_delay * self.environment.sound_speed
        
        # 物理范围约束
        max_dist = np.sqrt(self.environment.room_size[0]**2 + self.environment.room_size[1]**2)
        estimated_dist = np.clip(estimated_dist, 0.5, max_dist*1.1)
        
        # 保存结果
        self.estimated_distances.append(estimated_dist)
        self.true_distances.append(true_dist)
        
        return estimated_dist, true_dist, source_pos


def visualize(environment, perception_system, source_pos):
    """可视化环境、障碍物、感知系统和声源"""
    plt.figure(figsize=(12, 10))  # 增大画布尺寸
    
    # 绘制房间
    room_x, room_y = environment.room_size
    plt.plot([0, room_x, room_x, 0, 0], [0, 0, room_y, room_y, 0], 'k-', linewidth=1.5)
    
    # 绘制障碍物
    obs_pos, obs_radius = environment.obstacle
    circle = plt.Circle(obs_pos, obs_radius, color='gray', alpha=0.5)
    plt.gca().add_patch(circle)
    
    # 绘制感知系统移动路径
    path = np.array(perception_system.history_positions)
    plt.plot(path[:, 0], path[:, 1], 'b.-', markersize=8, label='Perception Path')
    plt.scatter(path[-1, 0], path[-1, 1], c='blue', s=120, label='Current Position')
    
    # 绘制声源
    plt.scatter(source_pos[0], source_pos[1], c='red', s=120, label='Sound Source')
    
    # 绘制连接线
    plt.plot([path[-1, 0], source_pos[0]], [path[-1, 1], source_pos[1]], 'r--', alpha=0.3)
    
    # 添加信息标注
    is_obstructed = environment._is_obstructed(source_pos, path[-1])
    error = abs(perception_system.estimated_distances[-1] - perception_system.true_distances[-1])
    rel_error = error / perception_system.true_distances[-1] * 100
    
    plt.text(0.05, 0.95, 
             f"Obstructed: {is_obstructed}\n"
             f"True Dist: {perception_system.true_distances[-1]:.2f}m\n"
             f"Est Dist: {perception_system.estimated_distances[-1]:.2f}m\n"
             f"Error: {error:.2f}m ({rel_error:.1f}%)",
             transform=plt.gca().transAxes, va='top',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.title(f'Acoustic Perception in {room_x}x{room_y}m Space', fontsize=14)
    plt.xlabel('X position (m)', fontsize=12)
    plt.ylabel('Y position (m)', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def main():
    # 创建50x50环境
    env = AcousticEnvironment(room_size=(50, 50))
    
    # 创建感知系统（初始位置设为(5,5)）
    perception_system = AcousticPerceptionSystem(initial_pos=(5, 5), environment=env)
    
    # 生成声源信号
    source_signal = env.generate_sound()
    
    # 模拟15次移动和距离估计（增加次数适应更大空间）
    for i in range(15):
        # 移动感知系统
        perception_system.move_randomly(step_size=3.0)  # 增大步长
        
        # 估计距离
        estimated_dist, true_dist, source_pos = perception_system.estimate_distance(source_signal)
        
        print(f"Step {i+1}: Position: {perception_system.position}, "
              f"Estimated: {estimated_dist:.2f}m, "
              f"True: {true_dist:.2f}m, "
              f"Obstructed: {env._is_obstructed(source_pos, perception_system.position)}, "
              f"Error: {abs(estimated_dist-true_dist):.2f}m ({abs(estimated_dist-true_dist)/true_dist*100:.1f}%)")
        
        # 可视化
        visualize(env, perception_system, source_pos)
        
        time.sleep(0.3)  # 缩短间隔加快演示


if __name__ == "__main__":
    main()