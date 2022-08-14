import time
import cv2
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def img_detect(model, source, roi=0, roi_range=[0, 0, 0, 0], imgsz=640, device='0', conf_thres=0.5, iou_thres=0.45, classes=None, agnostic_nms=False, augment=True):
    # ROI截取区域坐标
    x1 = roi_range[0]
    x2 = roi_range[1]
    y1 = roi_range[2]
    y2 = roi_range[3]

    t0 = time.time()

    result_label = []       # 存储检测结果，包括种类和置信度
    device = select_device(device)      # 设置设备
    half = device.type != 'cpu'  # 有CUDA支持时使用半精度

    stride = int(model.stride.max())    # model stride
    imgsz = check_img_size(imgsz, s=stride)  # 验证输入尺寸大小，如果不符合要求则进行自动调整
    if half:
        model.half()  # to FP16

    # 选取ROI(感兴趣区域)进行截取
    if roi:
        img = cv2.imread(source, 1)
        height = img.shape[0]  # 获取图像的高度
        width = img.shape[1]  # 获取图像的宽度
        if x1 < x2 < width and y1 < y2 < height:
            img_cut = img[y1:y2, x1:x2]
            cv2.imwrite('img_chopped.jpg', img_cut)
        else:
            print('ROI截取尺寸有误')
            print(width)
            print(height)

    # Set Dataloader,若进行ROI截取则读取img_chopped.jpg,若不进行ROI截取则读取原图，即source
    if roi:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages('img_chopped.jpg', img_size=imgsz, stride=stride)  # 加载数据
    else:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(source, img_size=imgsz, stride=stride)     # 加载数据

    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names

    t1 = time.time()        # 开始检测的时间

    # Run inference

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 进行一次前置推理，检测程序能否正常运行

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference 推理
        t2 = time_synchronized()        # 正式开始推理的时间
        pred = model(img, augment=augment)[0]

        # Apply NMS 应用非极大值抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()        # 非极大值抑制结束的时间

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
                    # 标准化检测框信息，xywh分别代表检测框的中心点坐标和宽高，宽高均是绝对长度除以图片宽高的结果
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # print(xywh)
                    # 保存检测结果的种类和置信度，并在原图上加以绘制
                    label = f'{names[int(cls)]} {conf:.2f}'
                    result_label.append(label)
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)        # 画图部分转移至窗口业务代码

            t4 = time.time()        # 完成推理、检测和保存结果的时间
            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            print(f'{s}Done. ')     # 打印图片大小和检测结果
            print(f'(img processing: {t1 - t0:.3f}s)')
            print(f'(pre_inference: {t2 - t1:.3f}s)')
            print(f'(inference+NMS: {t3 - t2:.3f}s)')
            print(f'(total time: {t4 - t0:.3f}s)')

            time_statistics = [t1-t0, t2-t1, t3-t2, t4-t0]      # 各部分用时数据

            # 返回结果
            return im0, result_label, det, time_statistics
