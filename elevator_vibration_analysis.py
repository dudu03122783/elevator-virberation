import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ElevatorVibrationAnalyzer:
    def __init__(self, csv_file, fs=1600):
        """初始化电梯振动分析器"""
        self.csv_file = csv_file
        self.fs = fs  # 采样频率
        self.data = None
        self.time = None
        self.ax = None  # x轴加速度
        self.ay = None  # y轴加速度
        self.az = None  # z轴加速度
        self.vz = None  # z轴速度
        self.sz = None  # z轴位移
    
    def load_data(self):
        """加载CSV数据"""
        print(f"正在加载数据: {self.csv_file}")
        try:
            # 读取CSV文件
            self.data = pd.read_csv(self.csv_file)
            
            # 提取数据并进行预处理
            self.time = np.arange(len(self.data)) / self.fs  # 计算时间轴
            self.ax = self.data['ax'].values - np.mean(self.data['ax'].values)
            self.ay = self.data['ay'].values - np.mean(self.data['ay'].values)
            self.az = self.data['az'].values - np.mean(self.data['az'].values)
            
            print(f"数据加载完成，共 {len(self.data)} 个数据点")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def calculate_velocity(self):
        """计算z轴速度（通过积分）"""
        if self.az is None:
            print("请先加载数据")
            return False
        
        print("正在计算z轴速度...")
        # 使用梯形法则积分计算速度
        self.vz = integrate.cumulative_trapezoid(self.az / 100, self.time, initial=0)
        print("速度计算完成")
        return True
    
    def calculate_displacement(self):
        """计算z轴位移（通过对速度积分）"""
        if self.vz is None:
            print("请先计算速度")
            return False
        
        print("正在计算z轴位移...")
        # 使用梯形法则积分计算位移
        self.sz = integrate.cumulative_trapezoid(self.vz, self.time, initial=0)
        print("位移计算完成")
        return True
    
    def plot_time_domain(self, save=False):
        """绘制时域图"""
        if self.time is None or self.ax is None:
            print("请先加载数据")
            return False
        
        print("正在绘制时域图...")
        plt.figure(figsize=(12, 12))
        
        # 绘制x轴加速度
        plt.subplot(5, 1, 1)
        plt.plot(self.time, self.ax)
        plt.title(f'{self.csv_file.split("\\")[-1]} 时域分析')
        plt.ylabel('x轴加速度 (gals)')
        plt.grid(True)
        
        # 绘制y轴加速度
        plt.subplot(5, 1, 2)
        plt.plot(self.time, self.ay)
        plt.ylabel('y轴加速度 (gals)')
        plt.grid(True)
        
        # 绘制z轴加速度
        plt.subplot(5, 1, 3)
        plt.plot(self.time, self.az)
        plt.ylabel('z轴加速度 (gals)')
        plt.grid(True)
        
        # 绘制z轴速度
        plt.subplot(5, 1, 4)
        plt.plot(self.time, self.vz)
        plt.ylabel('z轴速度 (m/s)')
        plt.grid(True)
        
        # 绘制z轴位移
        plt.subplot(5, 1, 5)
        if self.sz is not None:
            plt.plot(self.time, self.sz)
            plt.ylabel('z轴位移 (m)')
        plt.xlabel('时间 (s)')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save:
            save_path = self.csv_file.replace('.csv', '_时域图.png')
            plt.savefig(save_path)
            print(f"时域图已保存至: {save_path}")
        else:
            plt.show()
        
        return True
    
    def perform_fft(self, t_start=10, n=8192):
        """执行FFT分析"""
        if self.time is None or self.ax is None:
            print("请先加载数据")
            return False
        
        print("正在执行FFT分析...")
        # 计算起始和结束索引
        start_idx = int(t_start * self.fs)
        end_idx = start_idx + n - 1
        
        # 确保索引在有效范围内
        if end_idx >= len(self.time):
            print(f"警告: 数据长度不足，调整分析窗口")
            end_idx = len(self.time) - 1
            n = end_idx - start_idx + 1
        
        # 提取分析数据
        ax_segment = self.ax[start_idx:end_idx+1]
        ay_segment = self.ay[start_idx:end_idx+1]
        az_segment = self.az[start_idx:end_idx+1]
        
        # 执行FFT
        px = np.fft.fft(ax_segment)
        py = np.fft.fft(ay_segment)
        pz = np.fft.fft(az_segment)
        
        # 计算幅度 - 匹配MATLAB的计算方法
        px2 = np.abs(px / n)
        py2 = np.abs(py / n)
        pz2 = np.abs(pz / n)
        
        # 单边频谱 - 只对中间频率点乘以2（匹配MATLAB的计算方法）
        px1 = px2[:n//2]
        px1[1:-1] = 2 * px1[1:-1]
        
        py1 = py2[:n//2]
        py1[1:-1] = 2 * py1[1:-1]
        
        pz1 = pz2[:n//2]
        pz1[1:-1] = 2 * pz1[1:-1]
        
        # 计算频率轴
        f = np.linspace(0, self.fs/2, n//2)
        
        return f, px1, py1, pz1
    
    def plot_frequency_domain(self, f, px1, py1, pz1, save=False):
        """绘制频域图"""
        print("正在绘制频域图...")
        plt.figure(figsize=(12, 12))
        
        # 绘制x轴频谱
        plt.subplot(5, 1, 1)
        plt.plot(f, px1)
        plt.title(f'{self.csv_file.split("\\")[-1]} 频域分析')
        plt.ylabel('x轴加速度 (gals)')
        plt.grid(True)
        
        # 绘制y轴频谱
        plt.subplot(5, 1, 2)
        plt.plot(f, py1)
        plt.ylabel('y轴加速度 (gals)')
        plt.grid(True)
        
        # 绘制z轴频谱
        plt.subplot(5, 1, 3)
        plt.plot(f, pz1)
        plt.ylabel('z轴加速度 (gals)')
        plt.grid(True)
        
        # 绘制z轴速度
        plt.subplot(5, 1, 4)
        plt.plot(self.time, self.vz)
        plt.ylabel('z轴速度 (m/s)')
        plt.grid(True)
        
        # 绘制z轴位移
        plt.subplot(5, 1, 5)
        if self.sz is not None:
            plt.plot(self.time, self.sz)
            plt.ylabel('z轴位移 (m)')
        plt.xlabel('频率 (Hz)')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save:
            save_path = self.csv_file.replace('.csv', '_频域图.png')
            plt.savefig(save_path)
            print(f"频域图已保存至: {save_path}")
        else:
            plt.show()
        
        return True
    
    def analyze(self, save_plots=False):
        """执行完整分析流程"""
        if not self.load_data():
            return False
        
        if not self.calculate_velocity():
            return False
        
        if not self.calculate_displacement():
            return False
        
        self.plot_time_domain(save=save_plots)
        
        f, px1, py1, pz1 = self.perform_fft()
        self.plot_frequency_domain(f, px1, py1, pz1, save=save_plots)
        
        print("分析完成！")
        return True


def main():
    """主函数"""
    print("电梯振动数据分析软件")
    print("="*50)
    
    # 获取CSV文件路径
    csv_file = "c:\\Users\\HP\\Desktop\\震动数据 VER4\\20251015112725_SMECMJ2-23160375_24MAL10-249-29_轿内.csv"
    
    # 创建分析器实例
    analyzer = ElevatorVibrationAnalyzer(csv_file)
    
    # 执行分析
    analyzer.analyze(save_plots=False)  # 设置为True可以保存图像


if __name__ == "__main__":
    main()