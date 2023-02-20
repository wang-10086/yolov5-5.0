'''
此程序用于数据集中大量图片的整理,所有的图片将按照顺序进行重命名
'''

import os

path = "C:/Users/17262/Desktop/datasets/red&green/"     # 存储图片的文件夹路径
filelist = os.listdir(path)             # 该文件夹下所有的文件（包括文件夹）

count = 10496  # 确定重命名后起始图片名

for file in filelist:   # 遍历所有文件
    Olddir = os.path.join(path , file)   # 原来的文件路径
    if os.path.isdir(Olddir):   # 如果是文件夹则跳过
        continue
    filename = os.path.splitext(file)[0]   # 文件名
    filetype = os.path.splitext(file)[1]   # 文件扩展名

    Newdir = os.path.join(path,str(count).zfill(5)+filetype)    # 用字符串函数zfill,以0补全所需位数,zfill()中的'6'表示图片名称为6位数字
    os.rename(Olddir,Newdir)    # 重命名
    count+=1
print('重命名完成')
