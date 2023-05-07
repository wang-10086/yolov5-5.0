'''
本程序用于对标注数据集的标签文件进行统计,绘制yolo格式标签的中心位置分布和宽高分布情况
'''

import matplotlib.pyplot as plt
import os


def get_classes(file_path):
    classes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            classes.append(line)

    return classes


def main(path, img_width, img_height):
    '''
    path: 标签文件夹路径
    img_width: 图片宽度
    img_height: 图片高度
    '''

    # 数据读取
    label_list = []
    files = os.listdir(path)
    for file in files:
        if file == 'classes.txt':
            file_path = os.path.join(path, file)
            classes = get_classes(file_path)
            nc = len(classes)
        else:
            file_path = os.path.join(path, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                sents = []
                for line in lines:
                    sents = line.split()
                    c = int(sents[0])
                    x = float(sents[1])
                    y = float(sents[2])
                    w = float(sents[3])
                    h = float(sents[4])
                    label = [c, x, y, w, h]
                    label_list.append(label)

    # 绘制中心点分布和宽高分布
    x_plot, y_plot, w_plot, h_plot = [], [], [], []
    num = [0] * nc

    for label in label_list:
        # 统计各标签数量
        current_class = int(label[0])
        num[current_class] = num[current_class] + 1

        # 统计标签中心点坐标和宽高
        x = label[1] * img_width
        y = img_height - label[2] * img_height
        w = label[3] * img_width
        h = label[4] * img_height
        x_plot.append(x)
        y_plot.append(y)
        w_plot.append(w)
        h_plot.append(h)

    total_labels_num = len(label_list)
    print('%d label(s) in datasets' % total_labels_num)
    print('nc: %d' % nc)
    mean_width = sum(w_plot)/len(w_plot)
    mean_height = sum(h_plot)/len(h_plot)
    print('mean width: %d' % mean_width)
    print('mean height: %d' % mean_height)

    # 检测类别数量分布
    plt.figure(1)
    plt.title('label_classes distribution(total labels: %d)' % total_labels_num, fontsize=13)
    plt.bar(classes, num)
    for a, b in zip(classes, num):  # 柱子上的数字显示
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=8)

    # 目标中心点位置分布
    plt.figure(2)
    plt.title('Objects center position', fontsize=13)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.xlim(0, img_width)
    plt.ylim(0, img_height)
    plt.scatter(x_plot, y_plot, s=4, alpha=0.3)

    # 目标宽高分布
    plt.figure(3)
    plt.title('Objects width and height', fontsize=13)
    plt.xlabel('width', fontsize=12)
    plt.ylabel('height', fontsize=12)
    plt.scatter(w_plot, h_plot, s=6, alpha=0.4)
    plt.show()


if __name__ == '__main__':
    path = "E:/Datasets/实地采集数据集/环铁试验线/处理后数据集/train_signal/labels"
    img_width = 1440
    img_height = 1080

    main(path, img_width, img_height)
