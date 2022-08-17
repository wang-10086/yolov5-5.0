'''
此程序用于图片格式的修改
！！当使用'ren *.png *.jpg'命令直接批量修改图片格式时,有可能会使图片不可用,导致在图片标注的过程中出错,此时需要使用此程序对原始图片进行格式修改
！！常用做法是: 在使用'ren *.png *.jpg'命令直接批量修改图片格式后,使用此程序对所有jpg图片再进行一遍处理,以免出现某些图片不可用的情况
'''

import os
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dir_origin_path = "C:/Users/17262/Desktop/c/"       # 存储图片的文件夹路径
dir_save_path   = "C:/Users/17262/Desktop/b/"       # 保存修改后图片的文件夹路径

img_names = os.listdir(dir_origin_path)
for img_name in tqdm(img_names):
    if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        image_path  = os.path.join(dir_origin_path, img_name)
        image       = Image.open(image_path)
        image       = image.convert('RGB')

        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)
        image.save(os.path.join(dir_save_path, img_name))

