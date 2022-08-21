from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import *
from UiMainwindow import Ui_MainWindow

import sys

from model_load import model_load
from video_detect_thread import VideoDetectThread
from image_detect_thread import ImageDetectThread

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
        self.image_thread = None
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
            self.image_thread = ImageDetectThread(model=model, conf_thres=self.horizontalSlider.value(),
                                                  iou_thres=self.horizontalSlider_2.value())
            self.image_thread.signal.connect(self.transmit)
            self.image_thread.start()

        # 视频检测
        elif self.radioButton_2.isChecked() == 1:
            self.video_thread = VideoDetectThread(model=model, conf_thres=self.horizontalSlider.value(),
                                                  iou_thres=self.horizontalSlider_2.value())
            self.video_thread.signal.connect(self.transmit)
            self.video_thread.start()

        # 实时检测
        elif self.radioButton_3.isChecked() == 1:
            print('')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywin = MyMainWindow()
    mywin.show()
    sys.exit(app.exec_())