from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import *

import sys
import cv2

from model_load import model_load
from UiMainwindow import Ui_MainWindow
from video_detect_thread import VideoDetectThread
from image_detect_thread import ImageDetectThread

global model


class MyMainWindow(QMainWindow, Ui_MainWindow):
    signal_1 = pyqtSignal(object)       # 接收信号,用于接收来自子线程的检测结果图片
    signal_2 = pyqtSignal(object)       # 接收信号,用于接受来自子线程的检测结果数据
    signal_3 = pyqtSignal(object)     # 发送信号,用于向子线程发送conf_thres
    signal_4 = pyqtSignal(object)     # 发送信号,用于向子线程发送iou_thres

    def __init__(self, parent=None):
        global model
        weights = 'signal.pt'
        device = '0'

        # 界面初始化
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.detect)
        self.image_thread = None        # 初始化图片检测线程
        self.video_thread = None        # 初始化视频检测线程

        # 加载模型
        model = model_load(weights, device=device)
        print('模型加载完成')

    def display(self, img):
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
            self.image_thread.signal.connect(self.display)
            self.image_thread.start()

        # 视频检测
        elif self.radioButton_2.isChecked() == 1:
            self.video_thread = VideoDetectThread(model=model, conf_thres=self.horizontalSlider.value(),
                                                  iou_thres=self.horizontalSlider_2.value())
            self.video_thread.signal.connect(self.display)
            self.video_thread.start()

        # 实时检测
        elif self.radioButton_3.isChecked() == 1:
            print('')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywin = MyMainWindow()
    mywin.show()
    sys.exit(app.exec_())