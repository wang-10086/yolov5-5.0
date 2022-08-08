import sys
import cv2
import os
import time
import torch
from numpy import random

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal
from Ui_mainwindow import Ui_MainWindow
from settings import Ui_Form

from img_detect import img_detect
from model_load import model_load
from utils.plots import plot_one_box, plot_one_box_new
from utils.general import xyxy2xywh

global model
global video_quit
global realtime_quit


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        global model
        global video_quit
        global realtime_quit
        video_quit = 0
        realtime_quit = 0
        weights = './runs/train/exp/weights/best.pt'
        device = '0'

        # 界面初始化
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.img_detect)
        self.pushButton_2.clicked.connect(self.video_detect)
        self.pushButton_3.clicked.connect(self.video_quit)
        self.pushButton_4.clicked.connect(self.realtime_quit)
        self.pushButton_5.clicked.connect(self.realtime_detect)
        self.actionSettings.triggered.connect(lambda: self.opensettings())
        self.settings = QWidget()
        self.ui2 = Ui_Form()
        self.ui2.setupUi(self.settings)
        # 加载模型
        model = model_load(weights, device=device)
        print('模型加载完成')

    # 图片检测按钮槽函数--->pushButton
    def img_detect(self):
        """
        这是图片检测部分的代码

        :conf_thres: 置信度阈值
        :iou_thres: IOU阈值
        :source: 检测图片路径
        :roi: 1表示进行ROI截取,0表示不进行ROI截取
        :x1, x2, y1, y2: 分别表示ROI截取位置的左上角横坐标、右下角横坐标、左上角纵坐标、右下角纵坐标,当 roi=0 时，该参数无意义
        """
        global model

        source = 'C:/Users/17262/Desktop/test.jpg'  # 检测图片的路径
        roi = self.actionIs_ROI.isChecked()  # 是否进行ROI截取
        conf_thres = self.doubleSpinBox.value()  # 置信度阈值
        iou_thres = self.doubleSpinBox_2.value()  # IOU阈值
        x1 = 400  # ROI左上角横坐标
        x2 = 1400  # ROI右下角横坐标
        y1 = 600  # ROI左上角纵坐标
        y2 = 1040  # ROI右下角纵坐标
        roi_range = [x1, x2, y1, y2]

        original_img = cv2.imread(source)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        tl = 3 or round(0.002 * (original_img.shape[0] + original_img.shape[1]) / 2) + 1  # ROI框的线宽

        # 检测部分
        t0 = time.time()  # 开始检测,并计时
        # 调用图片检测函数img_detect()进行检测，result[0]存储检测结果，是np.ndarray类型;result[1]存储检测种类和置信度，是一个字符串数组;
        # det存储检测结果的各项信息，包括检测框位置
        result = img_detect(model=model, source=source, roi=roi, roi_range=roi_range, conf_thres=conf_thres, iou_thres=iou_thres)

        # 绘图部分
        if roi:
            offset = [x1, y1]  # 如果进行ROI，则最终返回坐标需要经过一定偏置，即映射到原图上才能进行绘图
            cv2.rectangle(original_img, [x1, y1], [x2, y2], color=[0, 0, 255], thickness=tl, lineType=cv2.LINE_AA)
        else:
            offset = [0, 0]

        gn = torch.tensor(result[0].shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(result[2]):
            # 标准化检测框信息，xywh分别代表检测框的中心点坐标和宽高，宽高均是绝对长度除以图片宽高的结果
            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

            # 保存检测结果的种类和置信度，并在原图上加以绘制
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box_new(xyxy, original_img, label=label, color=colors[int(cls)], line_thickness=3, offset=offset)

        # 对绘制后得到的结果进行加工处理
        img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)  # RGB to BGR
        img_result = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)

        t1 = time.time()  # 检测完成的时间

        # 将检测结果的类别和置信度显示在label_6上
        s = ''  # 空字符串用于存储检测结果
        for label in result[1]:
            s = s + label + '\n'
        self.label_6.setText(s)

        # 将结果在label_5中显示出来
        map = QPixmap.fromImage(img_result)
        self.label_5.setPixmap(map)
        self.label_5.setScaledContents(True)

        # 显示检测时间
        detect_time = str(round(t1 - t0, 3)) + 's'  # 检测时间保留3位小数
        self.label_7.setText(detect_time)

        print(f'检测用时({t1 - t0:.3f}s)')

    # 视频检测按钮槽函数--->pushButton_2
    def video_detect(self):
        """
        这是视频检测部分的代码

        :conf_thres: 置信度阈值,在每次检测实时获取
        :iou_thres: IOU阈值,在每次检测前实时获取
        :fps: 期望输出帧率,可用于控制视频播放速度
        :detect_frequency: 检测频率,如detect_frequency=5,即每隔5帧进行一次检测
        :roi: 1表示进行ROI截取,0表示不进行ROI截取
        :x1, x2, y1, y2: 分别表示ROI截取位置的左上角横坐标、右下角横坐标、左上角纵坐标、右下角纵坐标,当 roi=0 时，该参数无意义
        """

        global model
        global video_quit

        x1 = 400
        x2 = 1400
        y1 = 600
        y2 = 1040

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        try:
            # 文件读取
            video_path = QFileDialog.getOpenFileName(self, '选择视频文件', '.', 'Image files (*.mp4)')[0]
            if video_path == '':
                raise FileNotFoundError  # 如果未选择文件，即video_path为空，则主动抛出异常
            capture = cv2.VideoCapture(video_path)  # 读取视频

            # 视频检测
            num = 0  # 用于检测计数
            while True:
                # 视频检测退出功能
                if video_quit:
                    video_quit = 0
                    print('退出')
                    break

                roi = self.actionIs_ROI.isChecked()  # 是否进行ROI截取
                roi_range = [x1, x2, y1, y2]
                conf_thres = self.doubleSpinBox_3.value()  # 置信度阈值
                iou_thres = self.doubleSpinBox_4.value()  # IOU阈值
                fps = self.spinBox.value()  # 期望输出帧率
                detect_frequency = self.spinBox_2.value()  # 检测频率，即每detect_frequency帧检测一次

                t0 = time.time()  # 开始检测的时间
                ref, frame = capture.read()  # 读取当前帧
                if not ref:
                    print('读取当前帧失败')
                    break

                # 进入路口区域增加注意力集中机制
                if 85 < num + 1 < 180:
                    print('已进入路口区域')
                    roi = 1     # 进入路口后强制进行ROI截取
                    roi_range = [300, 1500, 300, 1000]      # 路口ROI截取范围,区别于正常ROI截取范围

                if roi:
                    offset = [roi_range[0], roi_range[2]]
                    leftup_point = [roi_range[0], roi_range[2]]     # ROI截取区域左上角点坐标
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
                self.label_8.setPixmap(result_map)
                self.label_8.setScaledContents(True)

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

    # 这是打开系统参数设置按钮槽函数-->actionsettings
    def opensettings(self):
        # self.settings = QMainWindow()
        # ui2 = Ui_SettingsWindow()
        # ui2.setupUi(settings)
        self.settings.show()

    # 这是视频检测退出按钮槽函数--->pushbutton3
    def video_quit(self):
        global video_quit
        video_quit = 1

    # 这是实时检测退出按钮槽函数--->pushbutton4
    def realtime_quit(self):
        global realtime_quit
        realtime_quit = 1

    # 实时检测按钮槽函数--->pushButton_5
    def realtime_detect(self):
        """
        这是实时检测部分的代码

        :conf_thres: 置信度阈值,在每次检测实时获取
        :iou_thres: IOU阈值,在每次检测前实时获取
        :fps: 期望输出帧率,可用于控制视频播放速度
        :detect_frequency: 检测频率,如detect_frequency=5,即每隔5帧进行一次检测
        :roi: 1表示进行ROI截取,0表示不进行ROI截取
        :x1, x2, y1, y2: 分别表示ROI截取位置的左上角横坐标、右下角横坐标、左上角纵坐标、右下角纵坐标,当 roi=0 时，该参数无意义
        """

        global model
        global realtime_quit

        x1 = 400
        x2 = 1400
        y1 = 600
        y2 = 1040

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        try:
            # 读取摄像头
            if self.action0.isChecked() == 1:
                camera = 0
            elif self.action1.isChecked() == 1:
                camera = 1
            else:
                raise CameraNotFound

            capture = cv2.VideoCapture(camera)  # 读取摄像头
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置宽度
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置长度

            # 视频检测
            num = 0  # 用于检测计数
            while True:
                # 视频检测退出功能
                if realtime_quit:
                    realtime_quit = 0
                    print('退出')
                    break

                roi = self.actionIs_ROI.isChecked()  # 是否进行ROI截取
                roi_range = [x1, x2, y1, y2]
                conf_thres = self.doubleSpinBox_5.value()  # 置信度阈值
                iou_thres = self.doubleSpinBox_6.value()  # IOU阈值
                fps = self.spinBox_3.value()  # 期望输出帧率
                detect_frequency = self.spinBox_4.value()  # 检测频率，即每detect_frequency帧检测一次

                t0 = time.time()  # 开始检测的时间
                ref, frame = capture.read()  # 读取当前帧
                if not ref:
                    print('读取当前帧失败')
                    break

                # 进入路口区域增加注意力集中机制
                # if 85 < num + 1 < 180:
                #     print('已进入路口区域')
                #     roi = 1  # 进入路口后强制进行ROI截取
                #     roi_range = [300, 1500, 300, 1000]  # 路口ROI截取范围,区别于正常ROI截取范围

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
                self.label_15.setPixmap(result_map)
                self.label_15.setScaledContents(True)

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
            # os.remove('original_img.jpg')
            # os.remove('img_chopped.jpg')

        except CameraNotFound:
            print('未选中任何摄像头，请选择一个摄像头')


class SettingsWindow(QWidget, Ui_Form):
    # mySignal = pyqtSignal(int)

    def __init__(self):
        super().__init__()


class CameraNotFound(Exception):
    pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywin = MyMainWindow()
    mywin.show()
    sys.exit(app.exec_())
