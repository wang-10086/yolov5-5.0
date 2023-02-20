# 铁路信号机视频自动识别与仿真系统

![](http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-10-28_21-18-20.jpg)

## Background

本项目是一款基于Yolov5目标检测算法和Pyqt5的铁路信号机视频识别仿真系统，旨在利用Yolov5目标检测算法对铁路行车过程中的信号机进行自动识别，并将结果以可视化界面的形式传递给用户，从而缓解司机在行车过程中的压力，并提高行车安全性。

在yolov5官方源码的基础上，我们增加了pyqt5界面程序、模型加载程序，同时修改了yolov5自带的Dataloader以便于更快地读取数据；
整个检测界面包括三个部分功能，分别是图片检测、视频检测、摄像头实时检测，采用多线程技术设计，基本能够实现流畅准确地检测效果。

## Install

1. 安装CUDA和Cudnn

2. 安装对应版本的Pytorch

3. 配置环境依赖

   ```python
   git clone https://github.com/wang-10086/yolov5-5.0  # clone
   cd yolov5-5.0
   pip install -r requirements.txt	# install
   ```

> 本项目的环境配置流程和yolov5官方项目配置流程几乎一致，您可以参考 [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)；
>
> 但注意，如果您按照yolov5的官方流程配置环境，还需要安装Pyqt5以使用界面部分功能。
>
> ```python
> pip install pyqt5
> ```

## Usage

执行mainwindow.py，或在任何IDE中直接点击运行即可：

```python
python mainwindow.py
```
> 注意：如果您的GPU算力有限，我们建议在使用本程序时不要使用其他程序占用GPU（如使用obs stdio进行录屏），否则会由于算力不足造成程序崩溃

本项目包含图片检测、视频检测和摄像头实时检测三个功能模块，下面为项目演示视频：
[道路信号机视频自动识别与仿真系统演示视频](http://wang-typora.oss-cn-beijing.aliyuncs.com/img/铁路信号机视频自动识别仿真系统演示视频（终）.mp4)

Update：对铁路信号机进行检测的演示视频：[铁路信号机视频检测演示视频](http://wang-typora.oss-cn-beijing.aliyuncs.com/img/presentation(train)_23_02_20.mp4)

## Datasets

受条件所限，我们选择使用道路信号灯来代替铁路信号机作为检测对象，通过道路信号灯的实地拍摄，以获得相应的数据集；

1. 线路长度16.88km，包含35个路口，途经商业区、居民区、城郊开发区等多种场景，信号灯包含竖式、横式、立式、悬挂式、圆形、箭头、人形道等多种类别，经过剪辑得到一段时长12分03秒、完全由路口信号灯组成的行车视频：
   
   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-06-13_20-32-14.jpg" style="zoom:50%;" />
2. 经过视频转图片操作后，得到共21712张图片，分辨率统一为1920*1080，在这21712张图片中我们筛选出了4000张图片作为数据集，数据集的标签构成如下；

   ```python
   gcircle		# 绿色圆形信号灯
   gleft		# 绿色左转信号灯
   gright		# 绿色右转信号灯
   gup			# 绿色直行信号灯
   rcircle		# 红色圆形信号灯
   rleft		# 红色左转信号灯
   rright		# 红色右转信号灯
   rup			# 红色直行信号灯
   ycircle		# 黄色圆形信号灯
   yleft		# 黄色左转信号灯
   yright		# 黄色右转信号灯
   yup			# 黄色直行信号灯
   gperson		# 绿色行人信号灯
   rperson		# 红色行人信号灯
   ```

   > 如：下图中的信号灯就是gperson:

   <img src="http://wang-typora.oss-cn-beijing.aliyuncs.com/img/Snipaste_2022-07-13_17-52-15.jpg" style="zoom:70%;" />

3. 本次拍摄得到数据集仍有一定的局限性，拍摄过程均在白天晴朗条件下完成，未考虑不同天气和不同时间条件因素的影响，如早晨、傍晚、夜间、阴天、雨雪等。

## Performance

1. 模型性能指标：

   ![](http://wang-typora.oss-cn-beijing.aliyuncs.com/img/results221028.png)

2. 检测速度：

   在CPU Intel i5-9300H，GPU NVIDIA GeForce GTX 1650环境下，视频检测速度最高可达29帧/s。

## Maintainers
@[Akkkk](https://github.com/wang-10086)
@[ykxxx](https://github.com/ykxxxxxx)

## Contact us
非常欢迎您使用我们的项目进行测试，如果您在使用过程中遇到任何问题，可以通过以下方式联系我们：

@[kunw13520935425@163.com](kunw13520935425@163.com)



