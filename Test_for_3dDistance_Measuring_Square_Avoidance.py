import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.spatial import distance
from scipy.signal import chirp, correlate
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
class AcousticEnvironment:
    def __init__(self, room_size=(50, 50, 50), obstacle=None):
        """
        初始化三维声学环境
        :param room_size: (长, 宽, 高) 单位米
        :param obstacle: ((x,y,z), 边长)  # 修改为正方体定义
        """
        self.room_size = room_size
        # 修改为正方体障碍物（中心点，边长）
        self.obstacle = obstacle if obstacle else ((25, 25, 25), 8)  # 中心正方体障碍物
        self.sound_speed = 343  # 声速 (m/s)
        self.sampling_rate = 192000  # 采样率 (Hz)

    def generate_sound(self, duration=0.15, freq_range=(500, 6000)):
        """生成带汉宁窗的线性调频信号"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration), False)
        signal = chirp(t, f0=freq_range[0], f1=freq_range[1], t1=duration)
        window = np.hanning(len(signal))
        return (signal * window) / np.max(np.abs(signal))

    def _is_obstructed(self, source_pos, receiver_pos):
        """三维线段与正方体的碰撞检测"""
        obs_center, side_length = self.obstacle
        half_side = side_length / 2
        
        # 计算正方体的边界框
        min_bounds = np.array(obs_center) - half_side
        max_bounds = np.array(obs_center) + half_side
        
        # 三维射线/线段与AABB碰撞检测算法
        source_pos = np.array(source_pos)
        receiver_pos = np.array(receiver_pos)
        direction = receiver_pos - source_pos
        max_length = np.linalg.norm(direction)
        
        if max_length == 0:
            return False
            
        # 标准化方向向量
        direction = direction / max_length
        
        t_min = 0.0
        t_max = max_length
        
        # 在三个坐标轴上检查碰撞
        for i in range(3):
            if abs(direction[i]) < 1e-6:  # 平行于坐标平面
                if source_pos[i] < min_bounds[i] or source_pos[i] > max_bounds[i]:
                    return False
            else:
                t1 = (min_bounds[i] - source_pos[i]) / direction[i]
                t2 = (max_bounds[i] - source_pos[i]) / direction[i]
                
                if t1 > t2:
                    t1, t2 = t2, t1
                    
                if t1 > t_min:
                    t_min = t1
                if t2 < t_max:
                    t_max = t2
                    
                if t_min > t_max:
                    return False
                
        # 确保交点在线段范围内
        return t_min >= 0 and t_max >= 0 and t_min <= max_length and t_max <= max_length

    def propagate_sound(self, source_pos, receiver_pos, signal):
        """三维声传播模型（含绕射和衰减）"""
        true_dist = np.linalg.norm(np.array(source_pos) - np.array(receiver_pos))
        
        # 计算有效传播距离
        if self._is_obstructed(source_pos, receiver_pos):
            # 简化模型：绕射距离按直线距离乘以系数计算
            detour_dist = true_dist * 1.35  # 正方体绕射系数
            attenuation = 0.18 / (true_dist ** 1.6)  # 正方体衰减系数
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
    
    # 绘制房间框架
    room_x, room_y, room_z = environment.room_size
    for s in [0, 1]:
        for t in [0, 1]:
            ax.plot([0, room_x], [s*room_y, s*room_y], [t*room_z, t*room_z], 'k-', alpha=0.3)
            ax.plot([s*room_x, s*room_x], [0, room_y], [t*room_z, t*room_z], 'k-', alpha=0.3)
            ax.plot([s*room_x, s*room_x], [t*room_y, t*room_y], [0, room_z], 'k-', alpha=0.3)
    
    # 绘制障碍物（正方体）
    obs_center, side_length = environment.obstacle
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
    
    # 绘制移动路径
    path = np.array(perception_system.history_positions)
    ax.plot(path[:,0], path[:,1], path[:,2], color='#1E90FF', linestyle='-')
    
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

    # 绘制声源位置并添加标注
    sound_source = ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='#00FF00', s=1700)
    ax.text(source_pos[0], source_pos[1], source_pos[2]+0.5, "Sound Source", 
            fontsize=19, color='black', 
            horizontalalignment='center',   # 水平居中对齐
            verticalalignment='bottom',     # 垂直底部对齐（点上方）
            zorder=10)                      # 设置较高的zorder确保文本显示在顶部
    
    # 添加声源到Agent的连线
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
    ax.set_title('3D Acoustic Perception System with Cube Obstacle')
    
    # 设置坐标轴范围与房间尺寸一致
    ax.set_xlim(0, room_x)
    ax.set_ylim(0, room_y)
    ax.set_zlim(0, room_z)
    
    # 添加图例
    from matplotlib.patches import Rectangle
    proxy = [Rectangle((0,0),1,1,fc='gold', alpha=0.8),
             plt.Line2D([0], [0], color='#1E90FF', lw=2),
             plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.5),
             plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor='blue'),
             plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor='navy'),
             plt.Line2D([0], [0], marker='o', color='w', markersize=15, markerfacecolor='#00FF00')]
    
    ax.legend(proxy, ['Cube Obstacle', 'Agent Path', 'Signal Path', 'Start Point', 'End Point', 'Sound Source'],
              loc='lower left', fontsize=12)
    
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