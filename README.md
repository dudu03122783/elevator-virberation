# 电梯振动分析程序

这是一个用于分析电梯振动数据的Python程序，包含基础分析和图形界面两个部分。

## 功能特点

- 加载电梯振动CSV数据文件
- 计算三轴加速度数据
- 通过积分计算z轴速度和位移
- 执行FFT频谱分析
- 绘制时域图和频域图
- 提供交互式GUI界面进行数据分析

## 依赖要求

- Python 3.7+
- numpy
- pandas
- matplotlib
- scipy
- tkinter

## 使用说明

### 基础分析

运行 `elevator_vibration_analysis.py` 进行基础的数据分析。

### 图形界面

运行 `elevator_vibration_analysis_gui.py` 打开交互式图形界面。

## 文件说明

- `elevator_vibration_analysis.py`: 基础分析模块，包含核心分析功能
- `elevator_vibration_analysis_gui.py`: 图形界面前端，提供用户交互功能

## 示例数据

程序设计用于处理包含ax、ay、az三轴加速度数据的CSV文件。