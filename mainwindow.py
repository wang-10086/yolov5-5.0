import os.path

import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用pyqt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # pyqt5的画布
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from PyQt5 import QtWidgets
import numpy as np
from PyQt5.QtGui import QPixmap, QImage, QIcon, QMouseEvent, QCursor
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QPoint

import time
from openpyxl import Workbook
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import sys
import cv2
import tkinter as tk
from tkinter import filedialog

from my_window_effect import WindowEffect
from model_load import model_load
from UiMainwindow import Ui_MainWindow
from utils.datasets import LoadStreams, LoadImages, set_camera_quit
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

global model, weights, device_id, camera_id, quit_flag, pause_flag, conf_thres, iou_thres, detect_frequency, fps, play_speed, is_time_log, is_position_log
"""
程序用到的全局变量：
model:  检测用到的模型
weights:    检测使用的权重文件
device_id:  检测设备，'0'、'1'、'2'表示使用0、1、2号GPU，'cpu'表示使用cpu检测
camera_id:  摄像头编号，'0'、'1'分别表示0、1号摄像头
quit_flag:  退出检测标志，为1时退出检测
pause_flag: 暂停与恢复标志，为1时暂停检测,为0时恢复检测
conf_thres: 置信度阈值
iou_thres:  IOU阈值
detect_frequency:   检测频率，即每隔detect_frequency检测一次
fps:    检测帧率
play_speed: 播放倍速，最高支持8倍速
is_time_log: 是否开启检测用时记录, 为1则实时显示单帧检测用时
is_position_log: 是否开启检测目标位置记录, 为1则实时显示检测到目标的轨迹变化
"""


class MyMainWindow(QMainWindow, Ui_MainWindow):
    """
    程序主界面部分。
    """

    def __init__(self, parent=None):
        """
        程序初始化
        """
        global model, weights, device_id, camera_id, quit_flag, pause_flag, conf_thres, iou_thres, detect_frequency, fps, play_speed, is_time_log, is_position_log

        # 全局变量初始化
        quit_flag = 0
        pause_flag = 0
        weights = 'train_signal.pt'
        device_id = '0'
        camera_id = '0'
        play_speed = 1

        # 界面初始化
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # self.setWindowFlags(Qt.CustomizeWindowHint)     # 去除标题栏
        self.setContentsMargins(0, 0, 0, 0)
        self.label_14.setText(weights)
        self.label_15.setText('GPU[0]')
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))     # 使主窗口位于屏幕正中

        # # 亚克力效果,实现窗口磨砂
        # self.windowEffect = WindowEffect()
        # self.resize(1800, 985)
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # self.setStyleSheet("background:transparent")  # 必须用样式表使背景透明，别用 setAttribute(Qt.WA_TranslucentBackground)，不然界面会卡顿
        # self.windowEffect.setAcrylicEffect(int(self.winId()))

        # log坐标窗口初始化
        self.time_figure = plt.Figure()     # 创建检测用时figure
        self.time_figure.patch.set_facecolor('none')    # 设置figure背景颜色为'none'，否则后面的canvas将无法透明
        self.time_figure.subplots_adjust(left=0.05, bottom=0.15, right=0.99, top=0.95)  # 设置figure边距
        self.time_canvas = FigureCanvasQTAgg(self.time_figure)  # 创建检测用时canvas
        self.time_canvas.setStyleSheet("background-color:transparent;")     # 设置canvas样式表为透明
        self.verticalLayout.addWidget(self.time_canvas)     # 将canvas添加到垂直布局中
        self.position_figure = plt.Figure() # 创建检测目标位置figure，后续操作同上
        self.position_figure.patch.set_facecolor('none')
        self.position_canvas = FigureCanvasQTAgg(self.position_figure)
        self.position_canvas.setStyleSheet("background-color:transparent;")
        self.verticalLayout_2.addWidget(self.position_canvas)

        self.pushButton.clicked.connect(self.detect)
        self.pushButton_2.clicked.connect(self.quit)
        self.pushButton_3.clicked.connect(self.pause)
        self.pushButton_4.clicked.connect(self.change_weights)
        self.pushButton_5.clicked.connect(self.history_clear)
        self.pushButton_6.clicked.connect(self.slowdown)
        self.pushButton_7.clicked.connect(self.speedup)

        self.horizontalSlider.valueChanged.connect(self.refresh_conf_thres)
        self.horizontalSlider_2.valueChanged.connect(self.refresh_iou_thres)
        self.spinBox.valueChanged.connect(self.refresh_detect_frequency)
        self.spinBox_2.valueChanged.connect(self.refresh_fps)
        self.comboBox.currentTextChanged.connect(self.change_camera)
        self.comboBox_2.currentTextChanged.connect(self.change_device)
        self.checkBox.toggled.connect(self.change_time_log)
        self.checkBox_2.toggled.connect(self.change_position_log)

        # 鼠标参数实例初始化
        self.origin_y = None
        self.origin_x = None
        self.mouse_Y = None
        self.mouse_X = None
        self.move_flag = None

        # 线程初始化
        self.image_thread = None  # 初始化图片检测线程
        self.video_thread = None  # 初始化视频检测线程
        self.realtime_thread = None  # 初始化实时检测线程

        # 检测参数初始化
        conf_thres = self.horizontalSlider.value()
        iou_thres = self.horizontalSlider_2.value()
        detect_frequency = self.spinBox.value()
        fps = self.spinBox_2.value()
        is_time_log = self.checkBox.isChecked()
        is_position_log = self.checkBox_2.isChecked()

        # 模型初始化
        model = model_load(weights, device=device_id)
        self.print_ifo('模型加载完成')
        print('模型加载完成')

    def time_plot(self, time_ifo):
        """
        单帧检测用时显示函数，接收来自子线程的用时信息，并实时显示
        """
        total_frame = time_ifo[0]   # 视频总帧数
        time_log = time_ifo[1]      # 用时记录列表，存储目前各帧的检测用时
        try:
            ax_time = self.time_figure.gca()            # 获取time_figure的坐标区
            frame = np.arange(0, len(time_log), 1)
            ax_time.cla()                               # 清空当前坐标区
            ax_time.set_xlim([0, total_frame])
            # ax_time.set_ylim([0, 0.1])
            ax_time.plot(frame[1:], time_log[1:])       # 绘制除了第一帧以外的检测用时记录，因为第一帧的用时数据往往异常偏高
            self.time_canvas.draw()                     # 显示图片
        except Exception as e:
            print(e)

    def position_plot(self, position_ifo):
        """
        检测目标轨迹变化实时显示函数，接收来自子线程的检测目标位置数据，并实时显示
        """
        x_position = []     # 创建空列表，存储x轴坐标
        y_position = []     # 创建空列表，存储y轴坐标
        position_log = position_ifo[0]                  # 位置信息
        img_width = position_ifo[1]                     # 图像宽度
        img_height = position_ifo[2]                    # 图像高度
        for position in position_log:
            x_position.append(position[0])
            y_position.append(img_height-position[1])
        try:
            ax_position = self.position_figure.gca()    # 获取position_figure的坐标区
            ax_position.cla()                           # 清空当前坐标区
            ax_position.set_xlim([0, img_width])
            ax_position.set_ylim([0, img_height])
            ax_position.set_xlabel('x')
            ax_position.set_ylabel('y')
            ax_position.scatter(x_position, y_position, s=4, alpha=0.3)
            self.position_canvas.draw()
        except Exception as e:
            print(e)

    def mousePressEvent(self, evt):
        """
        鼠标按压事件，确定两个点的位置(鼠标第一次按下的点以及窗口当前所在的原始点)，其中:
        mouse_X 和 mouse_Y是鼠标按下的点，相对于整个桌面而言;
        origin_x 和 origin_y是窗口左上角的点位置，也是相对于整个桌面而言。
        """
        self.move_flag = True
        self.mouse_X = evt.globalX()
        self.mouse_Y = evt.globalY()
        self.origin_x = self.x()
        self.origin_y = self.y()

    def mouseMoveEvent(self, evt):
        """
        鼠标移动事件:
        evt.globalX()和evt.globalY()是鼠标实时位置，由此可以计算出窗口需要移动的距离move_x 和 move_y，
        最终得出窗口移动目标点位置des_x 和 des_y
        """
        if self.move_flag:
            move_x = evt.globalX() - self.mouse_X
            move_y = evt.globalY() - self.mouse_Y
            des_x = self.origin_x + move_x
            des_y = self.origin_y + move_y
            self.move(des_x, des_y)

    def mouseReleaseEvent(self, QMouseEvent):
        """
        鼠标释放事件，将self.move_flag置为False
        """
        self.move_flag = False

    def display(self, img):
        """
        检测结果显示函数，接收来自子线程的检测结果图片，并在label加以显示
        """
        # 对绘制后得到的结果进行加工处理
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB to BGR
        img_result = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)

        # 将结果在label中显示出来
        map = QPixmap.fromImage(img_result)
        self.label.setPixmap(map)
        self.label.setScaledContents(True)

    def print_result(self, s):
        """
        检测结果输出函数，接收来自子线程的检测结果文本，并在label_6加以输出
        """
        self.label_6.setText(s)

    def print_ifo(self, s):
        """
        状态信息输出函数，在label_16加以输出
        """
        self.label_16.setText(s)

    def progress(self, progress):
        """
        进度条处理函数，接收视频检测子线程传回的当前帧和总帧数，在progressBar上实时显示处理进度
        """
        current_frame = progress[0]
        total_frame = progress[1]
        self.progressBar.setMaximum(total_frame)
        self.progressBar.setValue(current_frame)

    def quit(self):
        """
        退出检测函数，将quit_flag设为1，使得视频检测子线程或实时检测子线程终止
        """
        global quit_flag
        quit_flag = 1
        set_camera_quit(1)  # 将摄像头检测的退出标志quit_flag设为1
        self.pushButton_3.setIcon(QIcon(QPixmap('./resource/pause.png')))

    def pause(self):
        """
        暂停与恢复函数，将pause_flag置为1或0从而实现检测子线程挂起与恢复
        """
        global pause_flag

        if pause_flag == 1:
            self.print_ifo("线程已恢复")
            pause_flag = 0
            self.pushButton_3.setIcon(QIcon(QPixmap('./resource/pause.png')))
        else:
            self.print_ifo("线程已挂起")
            pause_flag = 1
            self.pushButton_3.setIcon(QIcon(QPixmap('./resource/resume.png')))

    def speedup(self):
        """
        加速播放函数，最高支持8倍速
        """
        global play_speed

        if 1 <= play_speed < 8:     # 最高支持8倍速
            play_speed = play_speed + 1
            self.print_ifo(str(play_speed) + '倍速')
        else:
            self.print_ifo('已达到最大播放速度')

    def slowdown(self):
        """
        减速播放函数，最高支持8倍速
        """
        global play_speed

        if 1 < play_speed <= 8:      # 最高支持8倍速
            play_speed = play_speed - 1
            self.print_ifo(str(play_speed)+'倍速')
        else:
            self.print_ifo('已达到最小播放速度')

    def history_clear(self):
        """
        清空上次检测的历史记录，包括显示画面、检测结果文本框、进度条以及log记录
        """
        self.label.setPixmap(QPixmap(""))
        self.label_6.clear()
        self.progressBar.setValue(0)
        ax_time = self.time_figure.gca()
        ax_time.cla()
        ax_position = self.position_figure.gca()
        ax_position.cla()

    def refresh_conf_thres(self, value):
        """
        刷新置信度阈值函数,作为horizontalSlider.valueChanged()信号的槽函数,其值改变时刷新conf_thres
        """
        global conf_thres
        conf_thres = value

    def refresh_iou_thres(self, value):
        """
        刷新IOU阈值函数,作为horizontalSlider_2.valueChanged()信号的槽函数,其值改变时刷新iou_thres
        """
        global iou_thres
        iou_thres = value

    def refresh_detect_frequency(self, value):
        """
        刷新检测频率函数,作为spinBox.valueChanged()信号的槽函数,其值改变时刷新detect_frequency
        """
        global detect_frequency
        detect_frequency = value

    def refresh_fps(self, value):
        """
        刷新输出帧率函数,作为spinBox_2.valueChanged()信号的槽函数,其值改变时刷新fps
        """
        global fps
        fps = value

    def change_time_log(self):
        """
        刷新检测用时记录标志
        """
        global is_time_log
        is_time_log = self.checkBox.isChecked()

    def change_position_log(self):
        """
        刷新目标位置记录标志
        """
        global is_position_log
        is_position_log = self.checkBox_2.isChecked()

    def change_camera(self, camera):
        """
        更改摄像头函数,作为comboBox.currentTextChanged()信号的槽函数,其值改变时改变camera_id
        """
        global camera_id

        test_cap = cv2.VideoCapture(int(camera))  # 创建一个VideoCapture对象以测试摄像头能否正常调用
        if test_cap is None or not test_cap.isOpened():
            self.print_ifo('打不开这个摄像头')
            print('Warning: unable to open camera.')
            test_cap.release()
        else:
            test_cap.release()
            camera_id = camera
            self.print_ifo('切换摄像头成功')
            print('Switch camera successfully.')

    def change_device(self, device):
        """
        更改检测设备函数,作为comboBox_2.currentTextChanged()信号的槽函数,其值改变时改变device_id
        """
        global model, weights, device_id
        device_id = device
        model = model_load(weights, device=device_id)
        if device_id == 'cpu':
            device_name = 'CPU'
        else:
            device_name = 'GPU[' + device_id + ']'
        self.label_15.setText(device_name)
        self.print_ifo('设备切换成功,模型已重新加载')
        print('设备切换成功,模型已重新加载')

    def change_weights(self):
        """
        更改权重文件函数,作为pushButton_4信号的槽函数,其值改变时改变权重文件weights
        """
        global model, device_id, weights

        # 实例化打开文件窗口
        root = tk.Tk()
        root.withdraw()

        try:
            weights = filedialog.askopenfilename(title='选择权重文件', filetypes=[('pt', '*.pt'), ('All files', '*')])
            model = model_load(weights, device=device_id)
            weights_name = os.path.basename(weights)
            self.label_14.setText(weights_name)
            self.print_ifo('权重切换成功,模型已重新加载')
            print('权重切换成功,模型已重新加载')

        except FileNotFoundError:
            self.print_ifo('请重新读取权重文件')
            print('请重新读取权重文件')

        root.destroy()
        root.mainloop()

    def detect(self):
        """
        检测函数,包括图片检测、视频检测、实时检测三个部分；
        检测部分均以多线程的方式进行。
        """
        global quit_flag, pause_flag, play_speed

        # 图片检测
        if self.radioButton.isChecked() == 1:
            self.image_thread = ImageDetectThread()
            self.image_thread.signal.connect(self.display)
            self.image_thread.signal2.connect(self.print_result)
            self.image_thread.start()

        # 视频检测
        elif self.radioButton_2.isChecked() == 1:
            # 每次检测前先重置一下quit_flag，防止因为被修改为1而无法正常检测
            quit_flag = 0
            pause_flag = 0
            play_speed = 1  # 重置播放速度为1倍速
            # 每次检测前清空Log坐标区
            ax_time = self.time_figure.gca()
            ax_time.cla()
            ax_position = self.position_figure.gca()
            ax_position.cla()
            self.pushButton_3.setIcon(QIcon(QPixmap('./resource/pause.png')))
            self.video_thread = VideoDetectThread()
            self.video_thread.signal.connect(self.display)
            self.video_thread.signal2.connect(self.print_result)
            self.video_thread.signal3.connect(self.progress)
            self.video_thread.signal4.connect(self.time_plot)
            self.video_thread.signal5.connect(self.position_plot)
            self.video_thread.start()

        # 实时检测
        elif self.radioButton_3.isChecked() == 1:
            # 每次检测前先重置一下quit_flag，防止因为被修改为1而无法正常检测
            quit_flag = 0
            pause_flag = 0
            self.pushButton_3.setIcon(QIcon(QPixmap('./resource/pause.png')))
            set_camera_quit(0)  # 将摄像头检测的datasets部分的quit_flag置为0,防止无法正常读取摄像头
            self.realtime_thread = RealtimeDetectThread()
            self.realtime_thread.signal.connect(self.display)
            self.realtime_thread.signal2.connect(self.print_result)
            self.realtime_thread.start()


class ImageDetectThread(QThread):
    _signal = pyqtSignal(object)  # 发送信号,用于向主线程发送检测结果图片
    _signal2 = pyqtSignal(object)  # 发送信号,用于向主线程发送检测结果数据

    def __init__(self, parent=None):
        super(ImageDetectThread, self).__init__(parent)

    def run(self):
        global model, device_id, quit_flag, conf_thres, iou_thres

        # 初始化
        result_label = []  # 空字符串存储检测结果
        imgsz = 640
        device = select_device(device_id)  # 设置设备
        half = device.type != 'cpu'  # 有CUDA支持时使用半精度

        # 实例化打开文件窗口
        root = tk.Tk()
        root.withdraw()

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # 验证输入尺寸大小，如果不符合要求则进行自动调整
        if half:
            model.half()  # to FP16

        try:
            # 文件读取
            image_path = filedialog.askopenfilename(title='选择图片', filetypes=[('Image', '*.jpg'), ('All files', '*')])
            if image_path == '':
                raise FileNotFoundError  # 如果未选择文件，即video_path为空，则主动抛出异常

            t0 = time.time()  # 开始检测时间
            # Set Dataloader
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadImages(image_path, img_size=imgsz, stride=stride)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            for path, img, im0s, vid_cap, current_frame, total_frame in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=True)[0]  # augment默认为True,后续可根据要求更改

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres / 100, iou_thres / 100, classes=None,
                                           agnostic=False)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            result_label.append(label)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                self._signal.emit(im0)

                # 将检测结果的类别和置信度返回
                s = ''  # 空字符串用于存储检测结果
                for label in result_label:
                    s = s + label + '\n'
                self._signal2.emit(s)

                t3 = time.time()  # 结束检测时间
                print(f'{s}Inference+NMS: ({t2 - t1:.3f}s)')
                print(f'总时长({t3 - t0:.3f}s)')

                root.mainloop()

        except FileNotFoundError:
            print('请重新读取文件')
            root.mainloop()

    @property
    def signal(self):
        return self._signal

    @property
    def signal2(self):
        return self._signal2


class VideoDetectThread(QThread):
    _signal = pyqtSignal(object)  # 发送信号,用于向主线程发送检测结果图片
    _signal2 = pyqtSignal(object)  # 发送信号,用于向主线程发送检测结果数据
    _signal3 = pyqtSignal(object)  # 发送信号,用于向主线程发送进度
    _signal4 = pyqtSignal(object)  # 发送信号,用于向主线程发送检测用时
    _signal5 = pyqtSignal(object)  # 发送信号,用于向主线程发送检测目标位置

    def __init__(self, parent=None):
        super(VideoDetectThread, self).__init__(parent)

    def run(self):
        global model, device_id, quit_flag, pause_flag, conf_thres, iou_thres, detect_frequency, fps, play_speed, is_time_log, is_position_log

        # 初始化
        imgsz = 640
        device = select_device(device_id)  # 设置设备
        half = device.type != 'cpu'  # 有CUDA支持时使用半精度
        time_log = []  # 储存每帧的检测时间
        position_log = []   # 储存检测到目标的位置信息

        # 实例化打开文件窗口
        root = tk.Tk()
        root.withdraw()

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # 验证输入尺寸大小，如果不符合要求则进行自动调整
        if half:
            model.half()  # to FP16

        try:
            # 文件读取
            video_path = filedialog.askopenfilename(title='选择视频', filetypes=[('Video', '*.mp4'), ('All files', '*')])
            if video_path == '':
                raise FileNotFoundError  # 如果未选择文件，即video_path为空，则主动抛出异常

            # Set Dataloader
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadImages(video_path, img_size=imgsz, stride=stride)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            for path, img, im0s, vid_cap, current_frame, total_frame in dataset:

                img0_height = im0s.shape[0]     # 获取原图高度
                img0_width = im0s.shape[1]      # 获取原图宽度

                # 挂起与恢复线程
                while pause_flag == 1:
                    if quit_flag == 1:
                        break
                    self.sleep(1)

                # 终止线程
                if quit_flag == 1:
                    quit_flag = 0
                    pause_flag = 0
                    print('已退出')
                    break

                # 按倍速进行播放
                if current_frame % play_speed == 0:
                    t0 = time.time()  # 每帧开始检测时间
                    result_label = []
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=True)[0]  # augment默认为True,后续可根据要求更改

                    # Apply NMS
                    pred = non_max_suppression(pred, conf_thres / 100, iou_thres / 100, classes=None,
                                               agnostic=False)
                    t2 = time_synchronized()

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                position = [0, 0]
                                label = f'{names[int(cls)]} {conf:.2f}'
                                result_label.append(label)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                                position[0] = (int(xyxy[0]) + int(xyxy[2])) / 2
                                position[1] = (int(xyxy[1]) + int(xyxy[3])) / 2
                                position_log.append(position)

                    self._signal.emit(im0)

                    # 将检测结果的类别和置信度返回
                    s = ''  # 空字符串用于存储检测结果
                    for label in result_label:
                        s = s + label + '\n'
                    self._signal2.emit(s)
                    self._signal3.emit([current_frame, total_frame])

                    # 延时程序，达到指定时间后进入下一循环
                    while time.time() - t0 < 1 / fps:
                        time.sleep(0.0001)

                    t3 = time.time()  # 结束检测时间
                    print(f'{s}Inference+NMS:({t2 - t1:.3f}s)')
                    print(f'总用时({t3 - t0:.3f}s)')
                    time_log.append(t3 - t0)
                    # 根据log标志位来判断是否向主线程返回log信息
                    if is_time_log:
                        self._signal4.emit([total_frame, time_log])
                    if is_position_log:
                        self._signal5.emit([position_log, img0_width, img0_height])

            # time_log
            del (time_log[0])       # 删除第一帧异常时间数据
            time_excel = Workbook()
            time_excel_ws = time_excel.active
            time_excel_ws['A1'] = 'time'
            for time0 in time_log:
                time_excel_ws.append([time0])
            time_excel.save('./log/time_log.xlsx')
            print(f'平均每帧用时({sum(time_log) / len(time_log):.3f}s)')

            # position_log
            x_position = []
            y_position = []
            position_excel = Workbook()
            position_excel_ws = position_excel.active
            position_excel_ws['A1'] = 'x'
            position_excel_ws['B1'] = 'y'
            for position in position_log:
                x_position.append(position[0])
                y_position.append(img0_height - position[1])
                position_excel_ws.append(position)
            position_excel.save('./log/position_log.xlsx')
            plt.title('Objects center position', fontsize=13)
            plt.xlabel('x', fontsize=12)
            plt.ylabel('y', fontsize=12)
            plt.xlim(0, img0_width)
            plt.ylim(0, img0_height)
            plt.scatter(x_position, y_position, s=4, alpha=0.3)
            plt.savefig('./log/position_log.png')
            plt.cla()
            plt.clf()

            root.mainloop()

        except FileNotFoundError:
            print('请重新读取文件')
            root.mainloop()


    @property
    def signal(self):
        return self._signal

    @property
    def signal2(self):
        return self._signal2

    @property
    def signal3(self):
        return self._signal3

    @property
    def signal4(self):
        return self._signal4

    @property
    def signal5(self):
        return self._signal5


class RealtimeDetectThread(QThread):
    _signal = pyqtSignal(object)  # 发送信号,用于向主线程发送检测结果图片
    _signal2 = pyqtSignal(object)  # 发送信号,用于向主线程发送检测结果数据

    def __init__(self, parent=None):
        super(RealtimeDetectThread, self).__init__(parent)

    def run(self):
        global model, device_id, camera_id, quit_flag, pause_flag, conf_thres, iou_thres, detect_frequency, fps

        # 初始化
        imgsz = 640
        device = select_device(device_id)  # 设置设备
        half = device.type != 'cpu'  # 有CUDA支持时使用半精度

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # 验证输入尺寸大小，如果不符合要求则进行自动调整
        if half:
            model.half()  # to FP16

        try:
            # Set Dataloader
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(camera_id, img_size=imgsz, stride=stride)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            for path, img, im0s, vid_cap in dataset:

                # 挂起与恢复线程
                while pause_flag == 1:
                    if quit_flag == 1:
                        break
                    self.sleep(1)

                # 终止线程
                if quit_flag == 1:
                    quit_flag = 0
                    pause_flag = 0
                    print('已退出')
                    break

                t0 = time.time()  # 每帧开始检测时间
                result_label = []
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=True)[0]  # augment默认为True,后续可根据要求更改

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres / 100, iou_thres / 100, classes=None,
                                           agnostic=False)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            result_label.append(label)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                self._signal.emit(im0)

                # 将检测结果的类别和置信度返回
                s = ''  # 空字符串用于存储检测结果
                for label in result_label:
                    s = s + label + '\n'
                self._signal2.emit(s)

                # 延时程序，达到指定时间后进入下一循环
                while time.time() - t0 < 1 / fps:
                    time.sleep(0.0001)

                t3 = time.time()  # 结束检测时间
                # Print time (inference + NMS)
                print(f'{s}Inference+NMS:({t2 - t1:.3f}s)')
                print(f'总用时({t3 - t0:.3f}s)')

        except FileNotFoundError:
            print('请重新读取文件')

    @property
    def signal(self):
        return self._signal

    @property
    def signal2(self):
        return self._signal2


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywin = MyMainWindow()
    mywin.setMouseTracking(True)
    mywin.show()
    sys.exit(app.exec_())
