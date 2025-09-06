import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# VOC路径配置
VOCdevkit_path = 'VOCdevkit'
segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')

def load_file_list(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() + '.png' for line in lines]

def analyze_distribution(file_list, label):
    classes_nums = np.zeros([256], int)

    for name in tqdm(file_list, desc=f"Processing {label}"):
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            print(f"标签图不存在：{png_file_name}")
            continue
        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print(f"跳过非灰度图：{name}, shape = {np.shape(png)}")
            continue
        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print(f"\n像素类别统计 ({label}):")
    print('-' * 37)
    print("| %15s | %15s |" % ("Class", "Pixel Count"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)
    return classes_nums

# 加载两个训练列表
train_list_old = load_file_list(os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation/train.txt'))
train_list_new = load_file_list(os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation/val.txt'))

# 分析两个列表
old_dist = analyze_distribution(train_list_old, "train.txt")
new_dist = analyze_distribution(train_list_new, "val.txt")