import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.gridspec import GridSpec
from scipy.fft import fft
from scipy import integrate
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib import font_manager

# 确保matplotlib使用TkAgg后端
import matplotlib
matplotlib.use('TkAgg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 全局变量
loaded_df = None  # 存储加载的数据
current_file_path = None  # 当前文件路径
vz_data = None  # z轴速度数据
sz_data = None  # z轴位移数据
sampling_rate = 1600  # 采样频率

class ElevatorVibrationAnalyzer:
    """电梯振动分析器类，封装数据处理和计算逻辑"""
    def __init__(self, df, fs=1600):
        """初始化电梯振动分析器"""
        self.df = df
        self.fs = fs  # 采样频率
        self.time = np.arange(len(self.df)) / self.fs  # 计算时间轴
        self.ax = None  # x轴加速度
        self.ay = None  # y轴加速度
        self.az = None  # z轴加速度
        self.vz = None  # z轴速度
        self.sz = None  # z轴位移
    
    def preprocess_data(self):
        """预处理数据：去除均值"""
        self.ax = self.df['ax'].values - np.mean(self.df['ax'].values)
        self.ay = self.df['ay'].values - np.mean(self.df['ay'].values)
        self.az = self.df['az'].values - np.mean(self.df['az'].values)
    
    def calculate_velocity(self):
        """计算z轴速度（通过积分）"""
        if self.az is None:
            self.preprocess_data()
        # 使用梯形法则积分计算速度
        self.vz = integrate.cumulative_trapezoid(self.az / 100, self.time, initial=0)
        return self.vz
    
    def calculate_displacement(self):
        """计算z轴位移（通过对速度积分）"""
        if self.vz is None:
            self.calculate_velocity()
        # 使用梯形法则积分计算位移
        self.sz = integrate.cumulative_trapezoid(self.vz, self.time, initial=0)
        return self.sz
    
    def get_all_data(self):
        """获取所有处理后的数据"""
        if self.ax is None:
            self.preprocess_data()
        if self.vz is None:
            self.calculate_velocity()
        if self.sz is None:
            self.calculate_displacement()
        
        return {
            'time': self.time,
            'ax': self.ax,
            'ay': self.ay,
            'az': self.az,
            'vz': self.vz,
            'sz': self.sz
        }
    
    def perform_fft(self, data, n=None):
        """执行FFT分析"""
        if n is None:
            n = len(data)
        
        # 执行FFT
        fft_result = fft(data)
        freq = np.fft.fftfreq(n, d=1 / self.fs)[:n // 2]
        
        # 先计算双边频谱幅度
        magnitude = np.abs(fft_result[:n // 2]) / n
        
        # 然后对单边频谱的中间频率点乘以2（匹配MATLAB的计算方法）
        if len(magnitude) > 2:
            magnitude[1:-1] = 2 * magnitude[1:-1]
        elif len(magnitude) == 2:
            magnitude[1] = 2 * magnitude[1]
        
        # 过滤频率范围为1-200Hz
        mask = (freq >= 1) & (freq <= 200)
        freq = freq[mask]
        magnitude = magnitude[mask]
        
        return freq, magnitude

def load_data(file_path):
    """加载CSV文件数据"""
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None

def perform_fft(data, sampling_rate):
    """对数据进行FFT分析"""
    # 执行FFT
    n = len(data)
    fft_result = fft(data)
    freq = np.fft.fftfreq(n, d=1 / sampling_rate)[:n // 2]
    
    # 先计算双边频谱幅度
    magnitude = np.abs(fft_result[:n // 2]) / n
    
    # 然后对单边频谱的中间频率点乘以2（匹配MATLAB的计算方法）
    if len(magnitude) > 2:
        magnitude[1:-1] = 2 * magnitude[1:-1]
    elif len(magnitude) == 2:
        magnitude[1] = 2 * magnitude[1]
    
    # 过滤频率范围为1-200Hz
    mask = (freq >= 1) & (freq <= 200)
    freq = freq[mask]
    magnitude = magnitude[mask]
    
    return freq, magnitude

def single_axis_fft_analysis(df, sampling_rate, axis, time_point, window_size, analyzer=None):
    """对特定轴进行FFT分析并提供保存功能"""
    try:
        # 检查轴是否存在
        if axis not in df.columns:
            messagebox.showerror("错误", f"数据中未找到{axis}列")
            return
        
        # 获取指定时间点的数据窗口
        start_idx = int(time_point * sampling_rate)
        end_idx = start_idx + int(window_size * sampling_rate)
        
        # 检查索引是否超出范围
        if end_idx > len(df):
            messagebox.showerror("错误", "选择的时间点超出数据范围")
            return
        
        # 获取数据
        data_window = df[axis].values[start_idx:end_idx]
        
        # 如果是速度或位移，需要从分析器获取
        if axis == 'vz' and analyzer:
            data_window = analyzer.get_all_data()['vz'][start_idx:end_idx]
        elif axis == 'sz' and analyzer:
            data_window = analyzer.get_all_data()['sz'][start_idx:end_idx]
        
        # 进行FFT分析
        freq, magnitude = perform_fft(data_window, sampling_rate)
        
        # 创建新的图形
        fig = plt.figure(figsize=(10, 6))
        
        # 设置颜色
        colors = {'ax': 'red', 'ay': 'green', 'az': 'blue', 'vz': 'purple', 'sz': 'orange'}
        color = colors.get(axis, 'black')
        
        # 设置标签
        labels = {'ax': 'x轴加速度', 'ay': 'y轴加速度', 'az': 'z轴加速度', 'vz': 'z轴速度', 'sz': 'z轴位移'}
        units = {'ax': 'gals', 'ay': 'gals', 'az': 'gals', 'vz': 'm/s', 'sz': 'm'}
        
        # 绘制FFT图
        plt.plot(freq, magnitude, color=color, linewidth=1.0)
        
        # 找到最大幅值及其对应的频率
        max_idx = np.argmax(magnitude)
        max_freq = freq[max_idx]
        max_magnitude = magnitude[max_idx]
        
        # 标记最大幅值点
        plt.scatter(max_freq, max_magnitude, color='black', s=50, zorder=5)
        # 添加标注文本，显示最大幅值和对应的频率
        plt.annotate(f'{max_magnitude:.2f} {units[axis]}\n{max_freq:.2f} Hz',
                    xy=(max_freq, max_magnitude),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        
        plt.title(f"{labels[axis]} - FFT分析 ({time_point:.2f}s到{time_point+window_size:.2f}s, {window_size}s窗口)")
        plt.xlabel("频率 (Hz)")
        plt.ylabel(f"幅值 ({units[axis]})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 显示图形
        plt.show(block=False)
        
        # 询问是否保存图片
        save_result = messagebox.askyesno("保存图片", "是否保存FFT分析图片？")
        if save_result:
            # 询问保存路径
            save_path = filedialog.asksaveasfilename(
                title="保存FFT分析图片",
                filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("所有文件", "*.*")],
                defaultextension=".png")
            
            if save_path:
                fig.savefig(save_path, dpi=300)
                messagebox.showinfo("保存成功", f"图片已保存至: {save_path}")
        
    except Exception as e:
        messagebox.showerror("错误", f"分析时发生错误: {str(e)}")

def perform_rolling_fft(data, sampling_rate, window_size):
    """进行滑动FFT分析"""
    # 计算窗口点数
    window_points = int(window_size * sampling_rate)
    # 计算步长大小（这里设置为窗口的1/4，可以调整）
    step_size = window_points // 4

    # 准备存储结果的数组
    n_steps = (len(data) - window_points) // step_size + 1
    freq = np.fft.fftfreq(window_points, d=1 / sampling_rate)[:window_points // 2]
    spectrogram = np.zeros((n_steps, window_points // 2))

    # 进行滑动FFT
    for i in range(n_steps):
        start_idx = i * step_size
        end_idx = start_idx + window_points
        window_data = data[start_idx:end_idx]
        fft_result = fft(window_data)
        
        # 先计算双边频谱幅度
        magnitude = np.abs(fft_result[:window_points // 2]) / window_points
        
        # 然后对单边频谱的中间频率点乘以2（匹配MATLAB的计算方法）
        if len(magnitude) > 2:
            magnitude[1:-1] = 2 * magnitude[1:-1]
        elif len(magnitude) == 2:
            magnitude[1] = 2 * magnitude[1]
        
        spectrogram[i, :] = magnitude

    return freq, spectrogram

def plot_combined_data(time, ax, ay, az, vz, sz, sampling_rate):
    """绘制组合数据：加速度、速度和位移，支持在图形中动态选择第四个曲线和纵向指针线"""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import numpy as np
    
    # 颜色设置
    colors = {
        'ax': 'red',
        'ay': 'green',
        'az': 'blue',
        'vz': 'purple',
        'sz': 'orange'
    }
    
    # 创建一个大图，留出按钮空间
    fig = plt.figure(figsize=(15, 13))  # 增加高度以容纳按钮
    
    # 调整布局，为按钮留出空间并优化显示效果 - 减小垂直间距，优化空间利用
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.07, hspace=0.1)  # 减小hspace以紧密连接图表
    
    # 绘制x轴加速度
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(time, ax, color=colors['ax'], linewidth=0.1)  # 设置加粗线宽
    # 将标题移到ylabel内部右侧
    ax1.set_ylabel('x轴加速度 (gals)', fontsize=10, labelpad=5)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制y轴加速度
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(time, ay, color=colors['ay'], linewidth=0.1)  # 设置加粗线宽
    # 将标题移到ylabel内部右侧
    ax2.set_ylabel('y轴加速度 (gals)', fontsize=10, labelpad=5)
    ax2.grid(True, linestyle='--', alpha=0.7)
    # 隐藏x轴刻度标签，因为共享x轴
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # 绘制z轴加速度
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(time, az, color=colors['az'], linewidth=0.1)  # 设置加粗线宽
    # 将标题移到ylabel内部右侧
    ax3.set_ylabel('z轴加速度 (gals)', fontsize=10, labelpad=5)
    ax3.grid(True, linestyle='--', alpha=0.7)
    # 隐藏x轴刻度标签，因为共享x轴
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    # 初始绘制速度曲线
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    line4, = ax4.plot(time, vz, color=colors['vz'], linewidth=2.0)  # 设置加粗线宽为2.0
    # 将标题移到ylabel内部右侧
    ax4.set_ylabel('z轴速度 (m/s)', fontsize=10, labelpad=5)
    ax4.set_xlabel('时间 (s)')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 存储所有子图的引用和相关数据
    axes_data = {
        ax1: {'data': ax, 'type': 'accel', 'unit': 'gals'},
        ax2: {'data': ay, 'type': 'accel', 'unit': 'gals'},
        ax3: {'data': az, 'type': 'accel', 'unit': 'gals'},
        ax4: {'data': vz, 'type': 'velocity', 'unit': 'm/s'}
    }
    
    # 初始化纵向指针线和数据标签
    vline_objects = {}
    text_objects = {}
    
    # 创建按钮轴
    ax_button_speed = plt.axes([0.35, 0.01, 0.15, 0.04])  # [left, bottom, width, height]
    ax_button_displacement = plt.axes([0.55, 0.01, 0.15, 0.04])
    
    # 创建按钮
    button_speed = Button(ax_button_speed, '显示z轴速度', color='#4CAF50', hovercolor='#45a049')
    button_displacement = Button(ax_button_displacement, '显示z轴位移', color='#f0f0f0', hovercolor='#e0e0e0')
    
    # 按钮回调函数
    def show_speed(event):
        line4.set_data(time, vz)
        # 将标题移到ylabel内部右侧
        ax4.set_ylabel('z轴速度 (m/s)', fontsize=10, labelpad=5)
        # 更新按钮颜色
        button_speed.color = '#4CAF50'
        button_speed.hovercolor = '#45a049'
        button_displacement.color = '#f0f0f0'
        button_displacement.hovercolor = '#e0e0e0'
        # 更新数据引用
        axes_data[ax4] = {'data': vz, 'type': 'velocity', 'unit': 'm/s'}
        # 自动调整y轴范围
        ax4.relim()
        ax4.autoscale_view()
        # 更新指针线标签（如果存在）
        for ax_obj in vline_objects:
            if ax_obj in text_objects:
                update_data_label(ax_obj, vline_objects[ax_obj].get_xdata()[0])
        # 重新绘制
        fig.canvas.draw_idle()
    
    def show_displacement(event):
        line4.set_data(time, sz)
        # 将标题移到ylabel内部右侧
        ax4.set_ylabel('z轴位移 (m)', fontsize=10, labelpad=5)
        # 更新按钮颜色
        button_speed.color = '#f0f0f0'
        button_speed.hovercolor = '#e0e0e0'
        button_displacement.color = '#4CAF50'
        button_displacement.hovercolor = '#45a049'
        # 更新数据引用
        axes_data[ax4] = {'data': sz, 'type': 'displacement', 'unit': 'm'}
        # 自动调整y轴范围
        ax4.relim()
        ax4.autoscale_view()
        # 更新指针线标签（如果存在）
        for ax_obj in vline_objects:
            if ax_obj in text_objects:
                update_data_label(ax_obj, vline_objects[ax_obj].get_xdata()[0])
        # 重新绘制
        fig.canvas.draw_idle()
    
    # 获取最近的数据点索引
    def get_nearest_index(x_val):
        return np.argmin(np.abs(time - x_val))
    
    # 更新单个子图的数据标签
    def update_data_label(ax_obj, x_val):
        if ax_obj not in axes_data or ax_obj not in text_objects:
            return
        
        data_info = axes_data[ax_obj]
        idx = get_nearest_index(x_val)
        y_val = data_info['data'][idx]
        
        # 根据数据类型设置不同的精度
        if data_info['type'] == 'accel':
            label_text = f't={x_val:.3f}s, 值={y_val:.4f} {data_info["unit"]}'
        elif data_info['type'] == 'velocity':
            label_text = f't={x_val:.3f}s, 值={y_val:.6f} {data_info["unit"]}'
        else:  # displacement
            label_text = f't={x_val:.3f}s, 值={y_val:.9f} {data_info["unit"]}'
        
        text_objects[ax_obj].set_text(label_text)
    
    # 鼠标点击事件处理函数
    def on_button_press(event):
        # 只有点击在子图上才处理
        if event.inaxes in axes_data:
            # 获取点击的x坐标
            x_val = event.xdata
            
            # 移除所有旧的指针线和标签
            for ax_obj in list(vline_objects.keys()):
                if ax_obj in vline_objects:
                    vline_objects[ax_obj].remove()
                if ax_obj in text_objects:
                    text_objects[ax_obj].remove()
            vline_objects.clear()
            text_objects.clear()
            
            # 为所有子图添加新的指针线和标签
            for ax_obj in axes_data.keys():
                # 添加纵向指针线
                vline = ax_obj.axvline(x=x_val, color='black', linestyle='--', alpha=0.7)
                vline_objects[ax_obj] = vline
                
                # 添加数据标签
                text = ax_obj.text(0.01, 0.95, '', transform=ax_obj.transAxes,
                                  bbox=dict(facecolor='white', alpha=0.8),
                                  fontsize=9)
                text_objects[ax_obj] = text
                
                # 更新标签内容
                update_data_label(ax_obj, x_val)
            
            # 重新绘制
            fig.canvas.draw_idle()
    
    # 连接按钮回调
    button_speed.on_clicked(show_speed)
    button_displacement.on_clicked(show_displacement)
    
    # 连接鼠标点击事件
    fig.canvas.mpl_connect('button_press_event', on_button_press)
    
    # 添加标题
    fig.suptitle('电梯振动数据组合分析', fontsize=16)
    
    # 使用一致的布局参数
    plt.show()

def plot_interactive_fft_analysis(df, sampling_rate):
    """交互式FFT分析，显示滑动窗口，集成速度和位移分析"""
    # 创建分析器实例
    analyzer = ElevatorVibrationAnalyzer(df, sampling_rate)
    analyzer_data = analyzer.get_all_data()
    
    # 初始设置
    window_size = 4  # 4秒窗口
    line_width = 0.25  # 初始线宽（适中粗细）
    current_time_point = 0
    current_axis = 'ax'  # 当前显示的轴
    
    # 创建图形，使用GridSpec设置布局
    fig = plt.figure(figsize=(18, 12))
    
    # 主显示区域的GridSpec
    main_gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # 菜单项状态变量
    window_size_var = 4
    time_position_var = 0
    line_width_var = 0.25  # 线宽设为0.25（适中粗细）
    
    # 滑动条变量
    slider = None
    
    # 颜色映射
    colors = {
        'ax': 'red',
        'ay': 'green',
        'az': 'blue',
        'vz': 'purple',
        'sz': 'orange'
    }
    
    # 标签映射
    labels = {
        'ax': 'x轴加速度',
        'ay': 'y轴加速度', 
        'az': 'z轴加速度',
        'vz': 'z轴速度',
        'sz': 'z轴位移'
    }
    
    # 单位映射
    units = {
        'ax': 'gals',
        'ay': 'gals',
        'az': 'gals',
        'vz': 'm/s',
        'sz': 'm'
    }
    
    # 存储所有图的图形对象
    ax_time_plots = []
    ax_fft_plots = []
    window_rects = []  # 存储窗口矩形
    fft_lines = []  # 存储FFT线条
    time_lines = []  # 存储时域线条
    
    # 创建用于显示坐标的文本框
    annot = None
    
    # 初始化时域数据和时间数组
    time = analyzer_data['time']
    
    def on_hover(event):
        nonlocal annot
        if event.inaxes:
            # 如果鼠标在任何坐标轴内
            ax = event.inaxes

            # 如果是时域图
            if ax in ax_time_plots:
                idx = ax_time_plots.index(ax)
                axis_names = ['ax', 'ay', 'az', 'vz', 'sz']
                if idx < len(axis_names):
                    axis_name = axis_names[idx]

                    # 获取最近的数据点
                    x_data = time
                    if axis_name in ['ax', 'ay', 'az']:
                        y_data = analyzer_data[axis_name]
                    else:
                        y_data = analyzer_data[axis_name]
                    
                    x_val = event.xdata
                    closest_idx = np.argmin(np.abs(x_data - x_val))
                    x_point = x_data[closest_idx]
                    y_point = y_data[closest_idx]

                    # 更新或创建注释
                    if annot is None:
                        annot = ax.annotate("",
                                            xy=(0, 0),
                                            xytext=(10, 10),
                                            textcoords="offset points",
                                            bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

                    # 设置注释位置为鼠标位置
                    annot.xy = (event.xdata, event.ydata)
                    text = f'时间: {x_point:.2f}s\n{labels[axis_name]}: {y_point:.2f} {units[axis_name]}'
                    annot.set_text(text)
                    annot.set_visible(True)

            # 如果是FFT图
            elif ax in ax_fft_plots:
                idx = ax_fft_plots.index(ax)
                axis_names = ['ax', 'ay', 'az', 'vz', 'sz']
                if idx < len(axis_names):
                    axis_name = axis_names[idx]

                    # 获取当前显示的FFT数据
                    line = fft_lines[idx]
                    x_data, y_data = line.get_data()
                    x_val = event.xdata
                    closest_idx = np.argmin(np.abs(x_data - x_val))
                    x_point = x_data[closest_idx]
                    y_point = y_data[closest_idx]

                    # 更新或创建注释
                    if annot is None:
                        annot = ax.annotate("",
                                            xy=(0, 0),
                                            xytext=(10, 10),
                                            textcoords="offset points",
                                            bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

                    # 设置注释位置为鼠标位置
                    annot.xy = (event.xdata, event.ydata)
                    text = f'频率: {x_point:.2f}Hz\n幅值: {y_point:.2f}'
                    annot.set_text(text)
                    annot.set_visible(True)

            fig.canvas.draw_idle()
        elif annot is not None:
            annot.set_visible(False)
            fig.canvas.draw_idle()
    
    # 连接鼠标移动事件
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    # 创建所有轴的子图
    all_axes = ['ax', 'ay', 'az', 'vz', 'sz']
    
    # 为每个轴创建子图（最多显示3个轴）
    for idx, axis in enumerate(all_axes[:3]):
        # 获取数据
        if axis in ['ax', 'ay', 'az']:
            data = analyzer_data[axis]
        else:
            data = analyzer_data[axis]
        
        # 时域图
        ax_time = fig.add_subplot(main_gs[idx, 0])
        line, = ax_time.plot(time, data, color=colors[axis], linewidth=line_width)
        time_lines.append(line)
        # 将标题信息移到ylabel中
        ax_time.set_ylabel(f'{labels[axis]} ({units[axis]})')
        if idx < 2:  # 只在最下方的图显示xlabel
            ax_time.set_xlabel('')
            plt.setp(ax_time.get_xticklabels(), visible=False)
        else:
            ax_time.set_xlabel('时间 (s)')
        ax_time.grid(True, linestyle='--', alpha=0.7)
        
        # 添加初始窗口矩形
        rect = plt.Rectangle((0, ax_time.get_ylim()[0]),
                             window_size,
                             ax_time.get_ylim()[1] - ax_time.get_ylim()[0],
                             facecolor='gray', alpha=0.2)
        ax_time.add_patch(rect)
        window_rects.append(rect)
        ax_time_plots.append(ax_time)
        
        # FFT图
        window_points = int(window_size * sampling_rate)
        ax_fft = fig.add_subplot(main_gs[idx, 1])
        freq_initial = np.fft.fftfreq(window_points, d=1 / sampling_rate)[:window_points // 2]
        mask = (freq_initial >= 1) & (freq_initial <= 200)
        freq_initial = freq_initial[mask]
        fft_initial = np.zeros_like(freq_initial)
        line, = ax_fft.plot(freq_initial, fft_initial, color=colors[axis], linewidth=line_width)
        fft_lines.append(line)
        
        # 将标题信息移到ylabel中
        ax_fft.set_ylabel(f'幅值 ({units[axis]})')
        if idx < 2:  # 只在最下方的图显示xlabel
            ax_fft.set_xlabel('')
            plt.setp(ax_fft.get_xticklabels(), visible=False)
        else:
            ax_fft.set_xlabel('频率 (Hz)')
        ax_fft.grid(True, linestyle='--', alpha=0.7)
        ax_fft.set_xlim(1, 200)
        ax_fft_plots.append(ax_fft)
    
    def update_fft(time_point):
        """更新FFT分析"""
        nonlocal current_time_point
        current_time_point = time_point
        
        window_points = int(window_size * sampling_rate)
        start_idx = int(time_point * sampling_rate)
        end_idx = start_idx + window_points

        # 确保索引不超出范围
        if end_idx > len(df):
            end_idx = len(df)
            start_idx = max(0, end_idx - window_points)

        # 更新窗口位置和FFT
        for idx, axis in enumerate(all_axes[:3]):
            # 更新窗口位置
            window_rects[idx].set_width(window_size)
            window_rects[idx].set_x(time_point)

            # 计算FFT
            if axis in ['ax', 'ay', 'az']:
                data_window = analyzer_data[axis][start_idx:end_idx]
            else:
                data_window = analyzer_data[axis][start_idx:end_idx]
                
            fft_result = fft(data_window)
            n = len(data_window)
            freq = np.fft.fftfreq(n, d=1 / sampling_rate)[:n // 2]
            
            # 先计算双边频谱幅度
            fft_magnitude = np.abs(fft_result[:n // 2]) / n
            
            # 然后对单边频谱的中间频率点乘以2（匹配MATLAB的计算方法）
            if len(fft_magnitude) > 2:
                fft_magnitude[1:-1] = 2 * fft_magnitude[1:-1]
            elif len(fft_magnitude) == 2:
                fft_magnitude[1] = 2 * fft_magnitude[1]

            # 过滤频率范围为1-200Hz
            mask = (freq >= 1) & (freq <= 200)
            freq = freq[mask]
            fft_magnitude = fft_magnitude[mask]

            # 更新FFT图
            fft_lines[idx].set_data(freq, fft_magnitude)
            
            # 清除之前的最大幅值标记
            for text in ax_fft_plots[idx].texts:
                text.remove()
            for scatter in ax_fft_plots[idx].collections:
                scatter.remove()
            
            # 找到最大幅值及其对应的频率
            if len(fft_magnitude) > 0:
                max_idx = np.argmax(fft_magnitude)
                max_freq = freq[max_idx]
                max_magnitude = fft_magnitude[max_idx]
                
                # 标记最大幅值点
                ax_fft_plots[idx].scatter(max_freq, max_magnitude, color='black', s=30, zorder=5)
                
                # 获取单位
                unit = units[axis]
                
                # 添加标注文本，显示最大幅值和对应的频率
                ax_fft_plots[idx].annotate(f'{max_magnitude:.2f} {unit}\n{max_freq:.2f} Hz',
                                        xy=(max_freq, max_magnitude),
                                        xytext=(10, 10),
                                        textcoords='offset points',
                                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                                        fontsize=8,
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            
            ax_fft_plots[idx].relim()
            ax_fft_plots[idx].autoscale_view()
            ax_fft_plots[idx].set_xlim(0, 200)

        fig.canvas.draw_idle()
    
    def update_line_width(val):
        """更新线宽"""
        nonlocal line_width
        line_width = val
        for line in time_lines:
            line.set_linewidth(line_width)
        fig.canvas.draw_idle()
    
    def update_window_size(label):
        """更新窗口大小并调整FFT计算"""
        nonlocal window_size
        window_size = float(label)
        
        # 更新窗口矩形
        for idx, axis in enumerate(all_axes[:3]):
            window_rects[idx].set_width(window_size)
        
        # 获取当前时间点
        current_time = current_time_point
        if slider is not None:
            max_time = len(df) / sampling_rate - window_size
            slider.valmax = max_time
            slider.ax.set_xlim(slider.valmin, slider.valmax)
            if slider.val > max_time:
                current_time = max_time
                slider.set_val(max_time)
            else:
                current_time = slider.val
        
        # 更新FFT显示
        update_fft(current_time)
    
    def update_displayed_axis(label):
        """更新显示的轴"""
        nonlocal current_axis
        current_axis = label
        # 这里可以添加轴切换逻辑
        fig.canvas.draw_idle()
    
    def on_single_axis_fft_analysis(event):
        """对选中的轴进行FFT分析"""
        selected_axis = current_axis
        single_axis_fft_analysis(df, sampling_rate, selected_axis, current_time_point, window_size, analyzer)
    
    def on_show_combined_data(event):
        """显示组合数据（加速度、速度、位移）"""
        plot_combined_data(
            analyzer_data['time'],
            analyzer_data['ax'],
            analyzer_data['ay'],
            analyzer_data['az'],
            analyzer_data['vz'],
            analyzer_data['sz'],
            sampling_rate
        )
    
    # 已移除toggle_control_panel函数，使用右键菜单替代侧边栏控制面板
    
    # 创建菜单函数
    def update_window_size_menu(size):
        """从菜单更新窗口大小"""
        nonlocal window_size, window_size_var
        window_size = float(size)
        window_size_var = float(size)
        update_window_size(size)
    
    def update_line_width_menu(width):
        """从菜单更新线宽"""
        nonlocal line_width, line_width_var
        line_width = float(width)
        line_width_var = float(width)
        update_line_width(line_width)
    
    def update_axis_menu(axis):
        """从菜单更新显示的轴"""
        nonlocal current_axis
        current_axis = axis
        update_displayed_axis(axis)
    
    def run_fft_analysis():
        """从菜单运行FFT分析"""
        single_axis_fft_analysis(df, sampling_rate, current_axis, current_time_point, window_size, analyzer)
    
    def show_combined_data_menu():
        """从菜单显示组合数据"""
        plot_combined_data(
            analyzer_data['time'],
            analyzer_data['ax'],
            analyzer_data['ay'],
            analyzer_data['az'],
            analyzer_data['vz'],
            analyzer_data['sz'],
            sampling_rate
        )
    
    def show_time_dialog(event=None):
         """显示时间设置对话框"""
         nonlocal current_time_point, time_position_var
         import tkinter as tk
         from tkinter import simpledialog
         
         # 创建Tk根窗口并隐藏
         root = tk.Tk()
         root.withdraw()
         
         # 获取最大时间
         max_time = len(df) / sampling_rate - window_size
         
         # 显示对话框
         new_time = simpledialog.askfloat("时间位置", f"输入时间位置 (0-{max_time:.2f}s):", 
                                        initialvalue=current_time_point, minvalue=0, maxvalue=max_time)
         
         if new_time is not None:
             current_time_point = new_time
             time_position_var = new_time
             # 如果滑动条存在，同步更新滑动条位置
             if 'slider_instance' in globals() and slider_instance is not None:
                 slider_instance.set_val(new_time)
             update_fft(new_time)
         
         root.destroy()
    
    def on_time_slider_changed(val):
         """处理时间滑动条变化事件"""
         nonlocal current_time_point, time_position_var
         try:
             current_time_point = float(val)
             time_position_var = float(val)
             if callable(update_fft):  # 检查update_fft是否可调用
                 update_fft(current_time_point)
         except Exception as e:
             print(f"Error in time slider: {e}")
    
    # 注意：不要创建新的figure，直接使用已有的fig
    menu_bar = fig.canvas.manager.toolbar
    
    # 添加右键菜单 - 优化实现，避免创建多个Tk实例和重复绑定事件
    def on_right_click(event):
        """处理右键点击事件，显示菜单"""
        import tkinter as tk
        from tkinter import Menu
        
        # 使用matplotlib的后端窗口位置信息
        fig_manager = plt.get_current_fig_manager()
        
        # 创建临时Tk根窗口但不进入主循环
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 创建弹出菜单
        popup = Menu(root, tearoff=0)
        
        # 添加窗口大小子菜单
        popup.add_command(label=f"窗口大小: {window_size}s", state="disabled")
        popup.add_separator()
        popup.add_command(label="窗口大小: 1s", command=lambda: (update_window_size_menu(1), root.destroy()))
        popup.add_command(label="窗口大小: 2s", command=lambda: (update_window_size_menu(2), root.destroy()))
        popup.add_command(label="窗口大小: 4s", command=lambda: (update_window_size_menu(4), root.destroy()))
        popup.add_command(label="窗口大小: 8s", command=lambda: (update_window_size_menu(8), root.destroy()))
        
        # 添加线宽子菜单
        popup.add_separator()
        popup.add_command(label=f"线宽: {line_width}", state="disabled")
        popup.add_separator()
        popup.add_command(label="线宽: 0.1", command=lambda: (update_line_width_menu(0.1), root.destroy()))
        popup.add_command(label="线宽: 0.5", command=lambda: (update_line_width_menu(0.5), root.destroy()))
        popup.add_command(label="线宽: 1.0", command=lambda: (update_line_width_menu(1.0), root.destroy()))
        popup.add_command(label="线宽: 2.0", command=lambda: (update_line_width_menu(2.0), root.destroy()))
        
        # 添加轴选择子菜单
        popup.add_separator()
        popup.add_command(label=f"当前轴: {current_axis}", state="disabled")
        popup.add_separator()
        popup.add_command(label="选择轴: ax", command=lambda: (update_axis_menu('ax'), root.destroy()))
        popup.add_command(label="选择轴: ay", command=lambda: (update_axis_menu('ay'), root.destroy()))
        popup.add_command(label="选择轴: az", command=lambda: (update_axis_menu('az'), root.destroy()))
        
        # 添加时间位置设置
        popup.add_separator()
        popup.add_command(label="设置时间位置...", command=lambda: (show_time_dialog(), root.destroy()))
        
        # 添加分析功能
        popup.add_separator()
        popup.add_command(label="当前轴FFT分析", command=lambda: (run_fft_analysis(), root.destroy()))
        popup.add_command(label="显示组合数据", command=lambda: (show_combined_data_menu(), root.destroy()))
        
        # 获取正确的屏幕坐标 - 适应不同的matplotlib后端
        try:
            if hasattr(fig_manager, 'window'):
                # 不同后端的窗口位置获取方式
                if hasattr(fig_manager.window, 'geometry') and callable(fig_manager.window.geometry):
                    # Tk后端
                    geo = fig_manager.window.geometry().split('+')
                    x_offset = int(geo[1])
                    y_offset = int(geo[2])
                elif hasattr(fig_manager.window, 'winfo_x') and hasattr(fig_manager.window, 'winfo_y'):
                    # GTK后端
                    x_offset = fig_manager.window.winfo_x()
                    y_offset = fig_manager.window.winfo_y()
                elif hasattr(fig_manager.window, 'x') and hasattr(fig_manager.window, 'y'):
                    # Qt后端
                    x_offset = fig_manager.window.x()
                    y_offset = fig_manager.window.y()
                else:
                    x_offset, y_offset = 0, 0
                
                # 计算菜单显示位置
                x = x_offset + event.x
                y = y_offset + event.y
            else:
                # 如果无法获取窗口位置，使用鼠标相对位置
                x, y = event.x, event.y
                
            # 显示菜单 - 使用Tk的post方法
            popup.post(x, y)
            
            # 为菜单绑定点击事件后自动销毁
            def on_popup_click(event=None):
                root.destroy()
            
            # 绑定菜单各项的选择事件
            for i in range(popup.index('end')+1):
                try:
                    cmd = popup.entrycget(i, 'command')
                    if cmd:
                        popup.entryconfigure(i, command=cmd)
                except:
                    pass
            
            # 确保菜单在点击后关闭
            root.after(100, lambda: root.destroy() if not popup.winfo_exists() else None)
            
            # 处理菜单事件，但不阻塞matplotlib事件循环
            root.update_idletasks()
            root.update()
            
        except Exception as e:
            # 如果菜单创建失败，输出提示信息
            print(f"创建右键菜单失败: {e}")
            # 添加一个备用的简单菜单实现
            try:
                # 使用matplotlib自身的菜单系统
                from matplotlib import rcParams
                rcParams['toolbar'] = 'toolbar2'  # 确保工具栏存在
                
                # 显示一个简单的提示文本
                fig.text(0.5, 0.5, "右键菜单创建失败，请尝试重新运行程序", 
                         fontsize=14, color='red', ha='center', va='center')
                fig.canvas.draw_idle()
                
            except:
                pass
    
    # 连接右键菜单事件 - 优化实现方式
    def handle_right_click(event):
        if event.button == 3:  # 右键点击
            on_right_click(event)
    
    fig.canvas.mpl_connect('button_press_event', handle_right_click)
    
    # 添加使用说明文本 - 调整位置避免与滑动条重叠
    fig.text(0.5, 0.01, '右键点击任意位置显示设置菜单', fontsize=10, color='red', ha='center')
    
    # 初始化FFT显示
    update_fft(0)
    
    # 创建时间滑动条 - 使用简单直接的方法
    max_time = len(df) / sampling_rate - window_size
    from matplotlib.widgets import Slider
    
    # 为滑动条留出足够空间
    plt.subplots_adjust(bottom=0.15)  # 增加底部边距为滑动条留出空间
    
    # 创建滑动条的轴位置
    slider_ax = fig.add_axes([0.2, 0.05, 0.65, 0.03])
    
    # 全局变量，确保回调函数能够访问
    global slider_instance
    slider_instance = Slider(
        ax=slider_ax,
        label='时间位置 (s)',
        valmin=0,
        valmax=max_time,
        valinit=0,
        valstep=0.1
    )
    
    # 直接定义一个简单的更新函数，避免嵌套函数问题
    def update_time_position(val):
        global current_time_point, time_position_var
        try:
            current_time_point = float(val)
            time_position_var = current_time_point  # 同步更新时间位置变量
            
            # 使用现有的update_fft函数来更新图表，确保一致性
            update_fft(current_time_point)
        except Exception as e:
            print(f"更新时间位置错误: {e}")
    
    # 连接滑动条事件 - 使用简单的函数引用
    slider_instance.on_changed(update_time_position)
    
    # 让图片最大限度占满整个屏幕，但为滑动条留出空间
    # 减小hspace使图表紧密连接，减小wspace优化左右间距
    plt.subplots_adjust(left=0.048, bottom=0.133, right=0.986, top=0.952, wspace=0.08, hspace=0.1)
    
    # 最大化显示窗口
    fig_manager = plt.get_current_fig_manager()
    try:
        # 不同的matplotlib后端可能有不同的最大化方法
        if hasattr(fig_manager, 'window'):
            if hasattr(fig_manager.window, 'state'):
                fig_manager.window.state('zoomed')  # GTK后端
            elif hasattr(fig_manager.window, 'maximize'):
                fig_manager.window.maximize()  # Tk后端
            elif hasattr(fig_manager.window, 'showMaximized'):
                fig_manager.window.showMaximized()  # Qt后端
    except Exception:
        # 如果最大化失败，至少调整subplots_adjust参数
        pass
    
    plt.show()

def show_custom_fft_dialog(df, sampling_rate):
    """显示自定义FFT分析对话框"""
    # 创建分析器实例
    analyzer = ElevatorVibrationAnalyzer(df, sampling_rate)
    analyzer.get_all_data()  # 确保计算了速度和位移
    
    # 创建对话框窗口
    dialog = tk.Toplevel()
    dialog.title("自定义FFT分析")
    dialog.geometry("400x350")
    dialog.resizable(False, False)
    
    # 创建轴选择
    tk.Label(dialog, text="选择分析轴：", font=("SimHei", 10)).grid(row=0, column=0, sticky="w", padx=20, pady=10)
    axis_var = tk.StringVar(value="ax")
    
    # 创建轴选择的框架
    axis_frame = tk.Frame(dialog)
    axis_frame.grid(row=0, column=1, padx=20, pady=10)
    
    # 添加轴选项，包括速度和位移
    tk.Radiobutton(axis_frame, text="x轴加速度", variable=axis_var, value="ax").pack(anchor="w")
    tk.Radiobutton(axis_frame, text="y轴加速度", variable=axis_var, value="ay").pack(anchor="w")
    tk.Radiobutton(axis_frame, text="z轴加速度", variable=axis_var, value="az").pack(anchor="w")
    tk.Radiobutton(axis_frame, text="z轴速度", variable=axis_var, value="vz").pack(anchor="w")
    tk.Radiobutton(axis_frame, text="z轴位移", variable=axis_var, value="sz").pack(anchor="w")
    
    # 创建时间点输入
    tk.Label(dialog, text="起始时间点 (s)：", font=("SimHei", 10)).grid(row=1, column=0, sticky="w", padx=20, pady=10)
    time_var = tk.StringVar(value="10")
    tk.Entry(dialog, textvariable=time_var, width=10).grid(row=1, column=1, sticky="w", padx=20, pady=10)
    
    # 创建窗口大小输入
    tk.Label(dialog, text="窗口大小 (s)：", font=("SimHei", 10)).grid(row=2, column=0, sticky="w", padx=20, pady=10)
    window_var = tk.StringVar(value="4")
    tk.Entry(dialog, textvariable=window_var, width=10).grid(row=2, column=1, sticky="w", padx=20, pady=10)
    
    def start_analysis():
        """开始分析"""
        try:
            axis = axis_var.get()
            time_point = float(time_var.get())
            window_size = float(window_var.get())
            
            # 检查时间范围
            max_time = len(df) / sampling_rate - window_size
            if time_point < 0 or time_point > max_time:
                messagebox.showerror("错误", f"起始时间应在0到{max_time:.2f}之间")
                return
            
            # 执行分析
            single_axis_fft_analysis(df, sampling_rate, axis, time_point, window_size, analyzer)
            dialog.destroy()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")
    
    # 创建按钮框架
    button_frame = tk.Frame(dialog)
    button_frame.grid(row=3, column=0, columnspan=2, pady=20)
    
    tk.Button(button_frame, text="开始分析", command=start_analysis, width=15, bg="lightblue").pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="取消", command=dialog.destroy, width=15, bg="lightcoral").pack(side=tk.LEFT, padx=10)
    
    # 居中显示窗口
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    dialog.grab_set()
    dialog.wait_window()

def main():
    """主程序"""
    # 创建主窗口
    root = tk.Tk()
    root.title("电梯振动数据分析软件")
    root.geometry("500x700")  # 增大窗口高度以容纳所有按钮
    
    # 设置全局变量
    global loaded_df, current_file_path, vz_data, sz_data
    
    # 创建状态变量
    status_var = tk.StringVar()
    status_var.set("请选择CSV数据文件")
    
    # 创建标题
    title_label = tk.Label(root, text="电梯振动数据分析软件", font=("SimHei", 16, "bold"))
    title_label.pack(pady=30)
    
    # 创建状态标签
    status_label = tk.Label(root, textvariable=status_var, font=("SimHei", 10))
    status_label.pack(pady=10)
    
    def select_file():
        """选择CSV文件"""
        global loaded_df, current_file_path
        
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(
            title="选择CSV数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            # 加载数据
            df = load_data(file_path)
            if df is not None:
                loaded_df = df
                current_file_path = file_path
                
                # 创建分析器并计算速度和位移
                analyzer = ElevatorVibrationAnalyzer(df, sampling_rate)
                analyzer.get_all_data()  # 计算所有数据
                
                # 更新状态
                status_var.set(f"已加载文件: {os.path.basename(file_path)}")
                # 启用分析按钮
                btn_interactive_fft.config(state=tk.NORMAL)
                btn_single_axis_fft.config(state=tk.NORMAL)
                btn_combined_data.config(state=tk.NORMAL)
                btn_global_fft.config(state=tk.NORMAL)
            else:
                loaded_df = None
                current_file_path = None
                # 禁用分析按钮
                btn_interactive_fft.config(state=tk.DISABLED)
                btn_single_axis_fft.config(state=tk.DISABLED)
                btn_combined_data.config(state=tk.DISABLED)
                btn_global_fft.config(state=tk.DISABLED)
    
    def show_interactive_fft():
        """显示交互式FFT分析"""
        if loaded_df is not None:
            try:
                plot_interactive_fft_analysis(loaded_df, sampling_rate)
                status_var.set("交互式FFT分析已启动")
            except Exception as e:
                status_var.set(f"交互式FFT分析出错: {str(e)}")
    
    def show_single_axis_fft():
        """显示单轴FFT分析"""
        if loaded_df is not None:
            try:
                show_custom_fft_dialog(loaded_df, sampling_rate)
            except Exception as e:
                status_var.set(f"单轴FFT分析出错: {str(e)}")
    
    def show_combined_data():
        """显示组合数据（加速度、速度、位移）"""
        if loaded_df is not None:
            try:
                # 创建分析器
                analyzer = ElevatorVibrationAnalyzer(loaded_df, sampling_rate)
                analyzer_data = analyzer.get_all_data()
                
                # 绘制组合数据
                plot_combined_data(
                    analyzer_data['time'],
                    analyzer_data['ax'],
                    analyzer_data['ay'],
                    analyzer_data['az'],
                    analyzer_data['vz'],
                    analyzer_data['sz'],
                    sampling_rate
                )
                status_var.set("组合数据显示已启动")
            except Exception as e:
                status_var.set(f"组合数据显示出错: {str(e)}")
    
    def show_global_fft():
        """显示全域FFT分析图"""
        if loaded_df is not None:
            try:
                # 创建分析器
                analyzer = ElevatorVibrationAnalyzer(loaded_df, sampling_rate)
                analyzer_data = analyzer.get_all_data()
                
                # 执行FFT分析
                time = analyzer_data['time']
                # 使用完整数据进行FFT分析
                n = len(analyzer_data['ax'])
                # 计算频率轴
                f = np.linspace(0, sampling_rate/2, n//2)
                
                # 对三个轴进行FFT分析
                def perform_full_fft(data):
                    # 执行FFT
                    fft_result = np.fft.fft(data)
                    # 计算幅度
                    magnitude = np.abs(fft_result / n)
                    # 单边频谱处理
                    magnitude = magnitude[:n//2]
                    if len(magnitude) > 2:
                        magnitude[1:-1] = 2 * magnitude[1:-1]
                    elif len(magnitude) == 2:
                        magnitude[1] = 2 * magnitude[1]
                    return magnitude
                
                px1 = perform_full_fft(analyzer_data['ax'])
                py1 = perform_full_fft(analyzer_data['ay'])
                pz1 = perform_full_fft(analyzer_data['az'])
                
                # 绘制频域图
                plt.figure(figsize=(12, 12))
                
                # 优化布局，减小垂直间距
                plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.07, hspace=0.1)
                
                # 添加整体标题
                plt.suptitle(f'{os.path.basename(current_file_path)} 频域分析', fontsize=14)
                
                # 绘制x轴频谱
                plt.subplot(5, 1, 1)
                plt.plot(f, px1, color='red', linewidth=0.1)  # 设置线宽为0.1
                plt.ylabel('x轴加速度 (gals)')
                plt.grid(True, linestyle='--', alpha=0.7)
                # 隐藏x轴刻度标签
                plt.setp(plt.gca().get_xticklabels(), visible=False)
                
                # 绘制y轴频谱
                plt.subplot(5, 1, 2)
                plt.plot(f, py1, color='green', linewidth=0.1)  # 设置线宽为0.1
                plt.ylabel('y轴加速度 (gals)')
                plt.grid(True, linestyle='--', alpha=0.7)
                # 隐藏x轴刻度标签
                plt.setp(plt.gca().get_xticklabels(), visible=False)
                
                # 绘制z轴频谱
                plt.subplot(5, 1, 3)
                plt.plot(f, pz1, color='blue', linewidth=0.1)  # 设置线宽为0.1        
                plt.ylabel('z轴加速度 (gals)')
                plt.grid(True, linestyle='--', alpha=0.7)
                # 隐藏x轴刻度标签
                plt.setp(plt.gca().get_xticklabels(), visible=False)
                
                # 绘制z轴速度
                plt.subplot(5, 1, 4)
                plt.plot(time, analyzer_data['vz'], color='purple', linewidth=0.5)  # 设置线宽为0.5
                plt.ylabel('z轴速度 (m/s)')
                plt.grid(True, linestyle='--', alpha=0.7)
                # 隐藏x轴刻度标签
                plt.setp(plt.gca().get_xticklabels(), visible=False)
                
                # 绘制z轴位移
                plt.subplot(5, 1, 5)
                plt.plot(time, analyzer_data['sz'], color='orange', linewidth=0.5)  # 设置线宽为0.5
                plt.ylabel('z轴位移 (m)')
                plt.xlabel('频率 (Hz) / 时间 (s)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()
                status_var.set("全域FFT分析已启动")
            except Exception as e:
                status_var.set(f"全域FFT分析出错: {str(e)}")
    
    def exit_program():
        """退出程序"""
        root.destroy()
    
    # 创建按钮框架
    button_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN)
    button_frame.pack(pady=40, fill=tk.X, padx=50)
    print("按钮框架已创建")
    
    # 创建按钮并设置样式 - 统一所有按钮的属性
    button_width = 25
    button_height = 2
    button_font = ('SimHei', 10)
    
    btn_select_file = tk.Button(button_frame, text="选择CSV数据文件", command=select_file,
                               width=button_width, height=button_height, bg="lightblue", font=button_font)
    btn_select_file.pack(pady=10)
    
    btn_interactive_fft = tk.Button(button_frame, text="交互式FFT分析", command=show_interactive_fft,
                                  width=button_width, height=button_height, state=tk.DISABLED, font=button_font)
    btn_interactive_fft.pack(pady=10)
    
    btn_single_axis_fft = tk.Button(button_frame, text="单轴FFT分析", command=show_single_axis_fft,
                                  width=button_width, height=button_height, state=tk.DISABLED, bg="lightgreen", font=button_font)
    btn_single_axis_fft.pack(pady=10)
    
    btn_combined_data = tk.Button(button_frame, text="显示组合数据", command=show_combined_data,
                                 width=button_width, height=button_height, state=tk.DISABLED, bg="lightyellow", font=button_font)
    btn_combined_data.pack(pady=10)
    
    btn_global_fft = tk.Button(button_frame, text="全域FFT图", command=show_global_fft,
                              width=button_width, height=button_height, state=tk.DISABLED, bg="lightpink", font=button_font)
    btn_global_fft.pack(pady=10)
    print("全域FFT图按钮已创建")
    
    btn_exit = tk.Button(button_frame, text="退出程序", command=exit_program,
                        width=button_width, height=button_height, bg="lightcoral", font=button_font)
    btn_exit.pack(pady=10)
    
    # 居中显示窗口
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    main()