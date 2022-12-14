'''
此程序的作用是找出在标签文件夹中没有相应的标签文件与之对应的图片,比如JPEGImages文件夹下存在001032和001312这两张图片,但是Annotations
文件夹中却没有001032和001312这两个标签文件与之对应,也就是说这两张图片是多余的,需要删除以构成图片文件与标签文件的一一对应关系。
'''

import os

path1 = "D:/python_project/object_detection/yolov5-u/signal/images/train2017/"      # path1为存储图片的文件夹JPEGImages
path2 = "D:/python_project/object_detection/yolov5-u/signal/labels/train2017/"     # path2为存储标签文件的文件夹Annotations

filelist1 = os.listdir(path1)   # 该文件夹下所有的文件（包括文件夹）
filelist2 = os.listdir(path2)   # 该文件夹下所有的文件（包括文件夹）

Illegalfiles = []    # 存放无法对应的图片文件名

for file1 in filelist1:   # 遍历所有文件
    count = 0
    filename1 = os.path.splitext(file1)[0]   # 文件名
    for file2 in filelist2:
        filename2 = os.path.splitext(file2)[0]   # 文件名
        if filename2 == filename1:
            count = 1
            break
    if count == 0:
        Illegalfiles.append(filename1)

## 输出结果
print('----------------------------------------')
if len(Illegalfiles) != 0:
    print('以下图片不存在与之对应的标签文件:')
    for Illegalfile in Illegalfiles:
        print(Illegalfile)
else:
    print('所有图片均存在与之对应的标签文件')
print('----------------------------------------')