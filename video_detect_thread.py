from PyQt5.QtCore import QThread, pyqtSignal

import time
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import tkinter as tk
from tkinter import filedialog

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class VideoDetectThread(QThread):
    _signal = pyqtSignal(object)        # 发送信号,用于向主线程发送检测结果图片
    _signal2 = pyqtSignal(object)       # 发送信号,用于向主线程发送检测结果数据

    def __init__(self, model, conf_thres, iou_thres, parent=None):
        super(VideoDetectThread, self).__init__(parent)
        self.model = model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def run(self):
        # 初始化
        model = self.model
        conf_thres = self.conf_thres / 100
        iou_thres = self.iou_thres / 100
        result_label = []  # 空字符串存储检测结果

        device = '0'
        imgsz = 640
        device = select_device(device)  # 设置设备
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
            for path, img, im0s, vid_cap in dataset:
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

                self._signal.emit(im0)

                # 将检测结果的类别和置信度显示在label_6上
                s = ''  # 空字符串用于存储检测结果
                for label in result_label:
                    s = s + label + '\n'
                self._signal2.emit(s)
                # self.label_6.setText(s)

                t3 = time.time()  # 结束检测时间
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                print(f'总用时({t3 - t0:.3f}s)')

                # time.sleep(5)

        except FileNotFoundError:
            print('请重新读取文件')

    @property
    def signal(self):
        return self._signal