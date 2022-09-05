from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal

import time
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import sys
import cv2
import tkinter as tk
from tkinter import filedialog

from model_load import model_load
from UiMainwindow import Ui_MainWindow
from utils.datasets import LoadStreams, LoadImages, set_camera_quit
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

global model, weights, device_id, camera_id, quit_flag, conf_thres, iou_thres, detect_frequency, fps
"""
程序用到的全局变量：
model:  检测用到的模型,程序内无需更改;
device_id:  检测设备,'0'、'1'、'2'表示使用0、1、2号GPU,'cpu'表示使用cpu检测
quit_flag:  退出检测标志,为1时退出检测,程序内无需更改
conf_thres: 置信度阈值,程序内无需更改
iou_thres:  IOU阈值,程序内无需更改
detect_frequency:   检测频率,即每隔detect_frequency检测一次,程序内无需更改
fps:    检测帧率,程序内无需更改
"""


class MyMainWindow(QMainWindow, Ui_MainWindow):
    signal_1 = pyqtSignal(object)  # 接收信号,用于接收来自子线程的检测结果图片
    signal_2 = pyqtSignal(object)  # 接收信号,用于接受来自子线程的检测结果数据
    signal_3 = pyqtSignal(object)  # 发送信号,用于向子线程发送conf_thres
    signal_4 = pyqtSignal(object)  # 发送信号,用于向子线程发送iou_thres

    def __init__(self, parent=None):
        """
        程序初始化
        """
        global model, weights, device_id, camera_id, quit_flag, conf_thres, iou_thres, detect_frequency, fps

        quit_flag = 0
        weights = 'C:/Users/17262/Desktop/coco128.pt'
        device_id = '0'
        camera_id = '0'

        # 界面初始化
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

        self.pushButton.clicked.connect(self.detect)
        self.pushButton_2.clicked.connect(self.quit)
        self.horizontalSlider.valueChanged.connect(self.refresh_conf_thres)
        self.horizontalSlider_2.valueChanged.connect(self.refresh_iou_thres)
        self.spinBox.valueChanged.connect(self.refresh_detect_frequency)
        self.spinBox_2.valueChanged.connect(self.refresh_fps)
        self.comboBox.currentTextChanged.connect(self.change_camera)
        self.comboBox_2.currentTextChanged.connect(self.change_device)
        self.pushButton_4.clicked.connect(self.change_weights)

        self.image_thread = None  # 初始化图片检测线程
        self.video_thread = None  # 初始化视频检测线程
        self.realtime_thread = None     # 初始化实时检测线程

        conf_thres = self.horizontalSlider.value()
        iou_thres = self.horizontalSlider_2.value()
        detect_frequency = self.spinBox.value()
        fps = self.spinBox_2.value()

        # 加载模型
        model = model_load(weights, device=device_id)
        print('模型加载完成')

    def display(self, img):
        """
        检测结果显示函数,接收来自子线程的检测结果图片,并在label加以显示
        """
        # 对绘制后得到的结果进行加工处理
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB to BGR
        img_result = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)

        # 将结果在label中显示出来
        map = QPixmap.fromImage(img_result)
        self.label.setPixmap(map)
        self.label.setScaledContents(True)

    def print_ifo(self, s):
        """
        检测结果输出函数,接收来自子线程的检测结果文本,并在label_6加以输出
        """
        self.label_6.setText(s)

    def progress(self, progress):
        """
        进度条处理函数,接收视频检测子线程传回的当前帧和总帧数,在progressBar上实时显示处理进度
        """
        current_frame = progress[0]
        total_frame = progress[1]
        self.progressBar.setMaximum(total_frame)
        self.progressBar.setValue(current_frame)

    def quit(self):
        """
        退出检测函数,将quit_flag设为1,使得视频检测子线程或实时检测子线程终止
        """
        global quit_flag
        quit_flag = 1
        set_camera_quit(1)      # 将摄像头检测的退出标志quit_flag设为1

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

    def change_camera(self, camera):
        """
        更改摄像头函数,作为comboBox.currentTextChanged()信号的槽函数,其值改变时改变camera_id
        """
        global camera_id

        test_cap = cv2.VideoCapture(int(camera))        # 创建一个VideoCapture对象以测试摄像头能否正常调用
        if test_cap is None or not test_cap.isOpened():
            print('Warning: unable to open camera.')
            test_cap.release()
        else:
            test_cap.release()
            camera_id = camera
            print('Switch camera successfully.')

    def change_device(self, device):
        """
        更改检测设备函数,作为comboBox_2.currentTextChanged()信号的槽函数,其值改变时改变device_id
        """
        global model, weights, device_id
        device_id = device
        model = model_load(weights, device=device_id)
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
            weights = filedialog.askopenfilename()
            model = model_load(weights, device=device_id)
            print('权重切换成功,模型已重新加载')

        except FileNotFoundError:
            print('请重新读取权重文件')

        root.mainloop()

    def detect(self):
        """
        检测函数,包括图片检测、视频检测、实时检测三个部分；
        检测部分均以多线程的方式进行。
        """
        global quit_flag

        # 图片检测
        if self.radioButton.isChecked() == 1:
            self.image_thread = ImageDetectThread()
            self.image_thread.signal.connect(self.display)
            self.image_thread.signal2.connect(self.print_ifo)
            self.image_thread.start()

        # 视频检测
        elif self.radioButton_2.isChecked() == 1:
            # 每次检测前先重置一下quit_flag，防止因为被修改为1而无法正常检测
            quit_flag = 0
            self.video_thread = VideoDetectThread()
            self.video_thread.signal.connect(self.display)
            self.video_thread.signal2.connect(self.print_ifo)
            self.video_thread.signal3.connect(self.progress)
            self.video_thread.start()

        # 实时检测
        elif self.radioButton_3.isChecked() == 1:
            # 每次检测前先重置一下quit_flag，防止因为被修改为1而无法正常检测
            quit_flag = 0
            set_camera_quit(0)
            self.realtime_thread = RealtimeDetectThread()
            self.realtime_thread.signal.connect(self.display)
            self.realtime_thread.signal2.connect(self.print_ifo)
            self.realtime_thread.start()


class ImageDetectThread(QThread):
    _signal = pyqtSignal(object)        # 发送信号,用于向主线程发送检测结果图片
    _signal2 = pyqtSignal(object)       # 发送信号,用于向主线程发送检测结果数据

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
            # image_path = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image files (*.jpg)')[0]
            image_path = filedialog.askopenfilename()
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
                pred = non_max_suppression(pred, conf_thres/100, iou_thres/100, classes=None,
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

    def __init__(self, parent=None):
        super(VideoDetectThread, self).__init__(parent)

    def run(self):
        global model, device_id, quit_flag, conf_thres, iou_thres, detect_frequency, fps

        # 初始化
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
            video_path = filedialog.askopenfilename()
            # video_path = 'C:/Users/17262/Desktop/test.mp4'
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
                # 终止线程
                if quit_flag == 1:
                    quit_flag = 0
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
                self._signal3.emit([current_frame, total_frame])

                # 延时程序，达到指定时间后进入下一循环
                while time.time() - t0 < 1 / fps:
                    time.sleep(0.0001)

                t3 = time.time()  # 结束检测时间
                # Print time (inference + NMS)
                print(f'{s}Inference+NMS:({t2 - t1:.3f}s)')
                print(f'总用时({t3 - t0:.3f}s)')

            root.mainloop()

        except FileNotFoundError:
            print('请重新读取文件')

    @property
    def signal(self):
        return self._signal

    @property
    def signal2(self):
        return self._signal2

    @property
    def signal3(self):
        return self._signal3


class RealtimeDetectThread(QThread):
    _signal = pyqtSignal(object)  # 发送信号,用于向主线程发送检测结果图片
    _signal2 = pyqtSignal(object)  # 发送信号,用于向主线程发送检测结果数据

    def __init__(self, parent=None):
        super(RealtimeDetectThread, self).__init__(parent)

    def run(self):
        global model, device_id, camera_id, quit_flag, conf_thres, iou_thres, detect_frequency, fps

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
                # 终止线程
                if quit_flag == 1:
                    quit_flag = 0
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
    mywin.show()
    sys.exit(app.exec_())
