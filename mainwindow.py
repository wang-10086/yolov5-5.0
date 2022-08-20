from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import *
from UiMainwindow import Ui_MainWindow

import sys

from model_load import model_load
from video_detect_thread import VideoDetectThread

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

global model


class MyMainWindow(QMainWindow, Ui_MainWindow):
    signal_1 = pyqtSignal(object)
    signal_2 = pyqtSignal(object)

    def __init__(self, parent=None):
        global model
        weights = 'signal.pt'
        device = '0'

        # 界面初始化
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.detect)
        self.video_thread = None
        # 加载模型
        model = model_load(weights, device=device)
        print('模型加载完成')

    def transmit(self, img):
        # 对绘制后得到的结果进行加工处理
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB to BGR
        img_result = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)

        # 将结果在label中显示出来
        map = QPixmap.fromImage(img_result)
        self.label.setPixmap(map)
        self.label.setScaledContents(True)

    def detect(self):
        global model
        # 图片检测
        if self.radioButton.isChecked() == 1:
            # 初始化
            conf_thres = self.horizontalSlider.value()/100
            iou_thres = self.horizontalSlider_2.value()/100
            result_label = []       # 空字符串存储检测结果

            device = '0'
            imgsz = 640
            device = select_device(device)  # 设置设备
            half = device.type != 'cpu'  # 有CUDA支持时使用半精度

            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # 验证输入尺寸大小，如果不符合要求则进行自动调整
            if half:
                model.half()  # to FP16

            try:
                # 文件读取
                # image_path = QFileDialog.getOpenFileName(self, '选择图像', '.', 'Image files (*.jpg)')[0]
                image_path = 'C:/Users/17262/Desktop/signal.jpg'
                if image_path == '':
                    raise FileNotFoundError  # 如果未选择文件，即video_path为空，则主动抛出异常

                t0 = time.time()    # 开始检测时间
                # Set Dataloader
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadImages(image_path, img_size=imgsz, stride=stride)

                # Get names and colors
                names = model.module.names if hasattr(model, 'module') else model.names
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

                # Run inference
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                for path, img, im0s, vid_cap in dataset:
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=True)[0]      # augment默认为True,后续可根据要求更改

                    # Apply NMS
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,
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

                    # 对绘制后得到的结果进行加工处理
                    img = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # RGB to BGR
                    img_result = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)

                    # 将结果在label中显示出来
                    map = QPixmap.fromImage(img_result)
                    self.label.setPixmap(map)
                    self.label.setScaledContents(True)

                    # 将检测结果的类别和置信度显示在label_6上
                    s = ''  # 空字符串用于存储检测结果
                    for label in result_label:
                        s = s + label + '\n'
                    self.label_6.setText(s)

                    t3 = time.time()    # 结束检测时间
                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')
                    print(f'({t3 - t0:.3f}s)')

            except FileNotFoundError:
                print('请重新读取文件')

        # 视频检测
        elif self.radioButton_2.isChecked() == 1:
            self.video_thread = VideoDetectThread(model=model, conf_thres=self.horizontalSlider.value(), iou_thres=self.horizontalSlider_2.value())
            self.video_thread._signal.connect(self.transmit)
            self.video_thread.start()

        # 实时检测
        elif self.radioButton_3.isChecked() == 1:
            print('')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywin = MyMainWindow()
    mywin.show()
    sys.exit(app.exec_())