# Yolo for Railway Signal-铁路信号机视频自动识别与仿真系统

<div align="center">
<img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/logo_23_04_25_20_59.png" style="zoom:80%;" />
</div>

🚀我们很高兴迎来了Yolo for Railway Signal的2.0版本，本项目自2022年7月正式开启以来，已经度过了9个月的时间。从最开始的和利时杯科技创新大赛的比赛项目，到后来成为我个人的本科毕业设计，我们从来都没有停止过对铁路信号机视频检测项目的探索和开发，希望我们的研究能够给您带来帮助。

## Background

本项目是一款基于Yolov5目标检测算法和Pyqt5的铁路信号机视频识别仿真系统，旨在利用Yolov5目标检测算法对铁路行车过程中的信号机进行自动识别，并将结果以可视化界面的形式传递给用户，从而缓解司机在行车过程中的压力，并提高行车安全性。

## Content

本项目的主要工作包含以下三个方面：

1. **改进Yolov5目标检测算法**：为了提高模型对小目标信号机的检测准确度，我们对Yolov5算法提出了三种改进策略，并通过消融实验进行了效果验证，三种策略分别如下:
   - mosaic9数据增强方法;
   - 带边缘扩展的copy-paste数据增强方法;
   - 增加小目标检测层。
2. **指示本列车运行的信号机的筛选分类模块**：该模块旨在实现对指示本列车运行的信号机的筛选分类，它由目标跟踪器DeepSort和多信号机筛选器MultiSignalFilter组成：
    - 目标跟踪器DeepSort：在Yolov5检测器的基础上使用DeepSort对信号机目标进行实时跟踪，得到信号机的连续变化轨迹；
    - 多信号机筛选器MultiSignalFilter：基于信号机的连续变化轨迹对多信号机进行筛选判断。
3. **铁路信号机视频自动识别与仿真系统**：基于PyQt5设计，具备图片检测、视频检测和实时检测三大功能，提供丰富的人机交互接口，支持实时调整检测参数，动态显示检测结果、单帧检测用时、信号机实时轨迹变化等。

## Install

1. 安装CUDA和Cudnn

2. 根据CUDA版本安装对应的Pytorch: [Installing previous versions of pytorch](https://pytorch.org/get-started/previous-versions/)

3. 配置环境依赖

   ```python
   git clone https://github.com/wang-10086/yolov5-5.0  # clone
   cd yolov5-5.0
   pip install -r requirements.txt	# install
   ```
   
   > 本项目使用Python=3.8环境，Pytorch版本为1.12.0

## Usage

<details>
<summary>运行</summary>

运行mainwindow.py：
```python
python mainwindow.py
```

</details>

<details>
<summary>界面介绍</summary>

整个界面包括参数设置模块、功能选择模块、结果显示模块、视频播放模块，同时还支持目标轨迹实时显示和检测用时实时显示模块。

<div align="center">
<img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2023-05-23_16-40-50.jpg" style="zoom:40%;" />
<br>
界面设计
</div>
</details>

<details>
<summary>功能演示</summary>

演示视频：[铁路信号机视频自动识别与仿真系统演示视频](http://wang-typora.oss-cn-beijing.aliyuncs.com/img/演示视频23_05_23.mp4 )

</details>

## Datasets

本项目在北京市朝阳区环铁试验线完成了铁路行车视频采集，数据集大小为10995张图片，数据集结构如下：

| 信号机种类 |  检测类别名称  |        说明        | 图片数量 |
|:-----:|:--------:|:----------------:|:----:|
|  红色   |   red    |    停车，禁止越过信号机    | 3100 |
|  绿色   |  green   |     允许越过该信号机     | 2800 |
|  单黄色  | s_yellow |    减速，或指示正线停车    | 1200 |
|  双黄色  | d_yellow |      指示侧线停车      | 2100 |
|  月白色  |  white   | 引导信号，或允许越过该信号机调车 | 1400 |
 |  蓝色   |   blue   |    禁止越过该信号机调车    | 895  |

## Checkpoints

|       model        |    datasets     | size | mAP<sup>optimal<br>50-95 |
|:------------------:|:---------------:|:----:|:------------------------:|
|     coco128.pt     |     coco128     | 640  |          0.7996          |
|   road_signal.pt   |   road_signal   | 640  |          0.7516          |
|  train_signal.pt   |  train_signal   | 640  |          0.7805          |
| train_signal500.pt | train_signal500 | 640  |          0.8204          |

## Deepsort

考虑到在信号机检测场景中，几乎不会出现短时间或长时间的目标重叠，也就是说无需考虑跟踪过程中的ID切换问题，本项目使用Deepsort作为目标跟踪器来对yolov5检测的结果进行实时跟踪，以获取同一个信号机的连续轨迹，进而根据轨迹变化特征进行信号机筛选。

<div align="center">
<img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/23_04_27_13_40.gif" alt="图片加载失败" width="70%" />
<br>
deepsort跟踪效果
</div>

本项目使用mikel.brostrom的[Yolov5 + Deep Sort with PyTorch](https://github.com/mikel-brostrom/yolov8_tracking/tree/v1.0 )。

## Maintainers
@[Akkkk](https://github.com/wang-10086)
@[ykxxx](https://github.com/ykxxxxxx)

## Contact us
非常欢迎您使用我们的项目进行测试，如果您在使用过程中遇到任何问题，可以通过以下方式联系我们：

[kunw13520935425@163.com](kunw13520935425@163.com)



