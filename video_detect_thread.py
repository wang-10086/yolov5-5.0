from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *

import cv2
import os
import time
import torch
from numpy import random

from img_detect import img_detect
from Ui_mainwindow import Ui_MainWindow
from utils.plots import plot_one_box, plot_one_box_new


class VideoDetectThread(QThread):
    signal = pyqtSignal(object)

    def __init__(self, model, parent=None):
        super(VideoDetectThread, self).__init__(parent)
        self.model = model

    def run(self):
        # for i in range(10):
        #     self.signal.emit(i)
        # self.signal.emit('程序结束')
        x1 = 400
        x2 = 1400
        y1 = 600
        y2 = 1040
        model = self.model

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        try:
            # 文件读取
            video_path = 'C:/Users/17262/Desktop/test.mp4'
            # video_path = QFileDialog.getOpenFileName(self, '选择视频文件', '.', 'Image files (*.mp4)')[0]
            if video_path == '':
                raise FileNotFoundError  # 如果未选择文件，即video_path为空，则主动抛出异常
            capture = cv2.VideoCapture(video_path)  # 读取视频

            # 视频检测
            num = 0  # 用于检测计数
            while True:
                # 视频检测退出功能
                # if video_quit:
                #     video_quit = 0
                #     print('退出')
                #     break

                roi = 0  # 是否进行ROI截取
                roi_range = [x1, x2, y1, y2]
                conf_thres = 0.5  # 置信度阈值
                iou_thres = 0.45  # IOU阈值
                fps = 30  # 期望输出帧率
                detect_frequency = 1  # 检测频率，即每detect_frequency帧检测一次
                # roi = Ui_MainWindow.actionIs_ROI.isChecked()  # 是否进行ROI截取
                # roi_range = [x1, x2, y1, y2]
                # conf_thres = Ui_MainWindow.doubleSpinBox_3.value()  # 置信度阈值
                # iou_thres = Ui_MainWindow.doubleSpinBox_4.value()  # IOU阈值
                # fps = Ui_MainWindow.spinBox.value()  # 期望输出帧率
                # detect_frequency = Ui_MainWindow.spinBox_2.value()  # 检测频率，即每detect_frequency帧检测一次

                t0 = time.time()  # 开始检测的时间
                ref, frame = capture.read()  # 读取当前帧
                if not ref:
                    print('读取当前帧失败')
                    break

                # 进入路口区域增加注意力集中机制
                if 85 < num + 1 < 180:
                    print('已进入路口区域')
                    roi = 1  # 进入路口后强制进行ROI截取
                    roi_range = [300, 1500, 300, 1000]  # 路口ROI截取范围,区别于正常ROI截取范围

                if roi:
                    offset = [roi_range[0], roi_range[2]]
                    leftup_point = [roi_range[0], roi_range[2]]  # ROI截取区域左上角点坐标
                    rightdown_point = [roi_range[1], roi_range[3]]  # ROI截取区域右下角点坐标
                else:
                    offset = [0, 0]

                cv2.imwrite('original_img.jpg', frame)  # 将当前帧暂存为jpg图像
                original_img = cv2.imread('original_img.jpg')
                tl = 3 or round(0.002 * (original_img.shape[0] + original_img.shape[1]) / 2) + 1  # ROI框的线宽

                # 进行检测，每隔detect_frequency帧进行一次检测
                if num % detect_frequency == 0:
                    # 检测部分
                    results = img_detect(model=model, source='original_img.jpg', roi=roi, roi_range=roi_range,
                                         conf_thres=conf_thres, iou_thres=iou_thres)

                    # 绘图部分
                    if roi:
                        cv2.rectangle(original_img, leftup_point, rightdown_point, color=[0, 0, 255], thickness=tl,
                                      lineType=cv2.LINE_AA)
                    gn = torch.tensor(results[0].shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    for *xyxy, conf, cls in reversed(results[2]):
                        # 标准化检测框信息，xywh分别代表检测框的中心点坐标和宽高，宽高均是绝对长度除以图片宽高的结果
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        # 保存检测结果的种类和置信度，并在原图上加以绘制
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box_new(xyxy, original_img, label=label, color=colors[int(cls)], line_thickness=3,
                                         offset=offset)

                    # RGB-->BGR，并转换为QImage格式
                    img_result = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                    img_result = QImage(img_result, img_result.shape[1], img_result.shape[0], img_result.shape[1] * 3,
                                        QImage.Format_RGB888)

                    # 将QImage转换为QPixmap
                    result_map = QPixmap.fromImage(img_result)

                else:
                    # 不进行检测时，直接输出原图像
                    result_map = QPixmap('original_img.jpg')

                # 将图片在label_8上进行显示
                # Ui_MainWindow.label_8.setPixmap(result_map)
                # Ui_MainWindow.label_8.setScaledContents(True)

                # 计数加一
                num = num + 1

                # 延时程序，达到指定时间后进入下一循环
                while time.time() - t0 < 1 / fps:
                    time.sleep(0.0001)
                t1 = time.time()
                print('第%d帧检测用时：%.4fs' % (num, t1 - t0))

                # 退出功能
                c = cv2.waitKey(0) & 0xff  # 判断按下按键
                if c == 27:  # 若按下ESC则退出
                    capture.release()
                    break

            # 释放capture，销毁所有窗口，删除临时文件
            capture.release()
            # cv2.destroyAllWindow()
            os.remove('original_img.jpg')
            os.remove('img_chopped.jpg')

        except FileNotFoundError:
            print('请重新读取文件')
        self.signal.emit('程序结束')