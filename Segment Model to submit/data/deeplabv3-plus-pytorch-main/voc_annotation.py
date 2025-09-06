import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#   
#   当前该库将测试集当作验证集使用，不单独划分测试集
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path      = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    file_list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(file_list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    from collections import defaultdict

# 构建每个类别对应的图像索引
    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

    # 统计图像与类的对应关系
    class_to_images = defaultdict(set)
    image_to_classes = defaultdict(set)
    total_seg = [f for f in os.listdir(segfilepath) if f.endswith(".png")]
    index_to_name = {i: name for i, name in enumerate(total_seg)}
    name_to_index = {v: k for k, v in index_to_name.items()}

    print("扫描所有标签以建立类别与图像之间的映射...")
    for i, name in enumerate(tqdm(total_seg)):
        png_file = os.path.join(segfilepath, name)
        png = np.array(Image.open(png_file), np.uint8)
        unique_labels = np.unique(png)
        for label in unique_labels:
            if 1 <= label <= 17:
                class_to_images[label].add(i)
                image_to_classes[i].add(label)

    # 用最多的图像覆盖所有类别（贪心思路）
    covered_classes = set()
    selected_indices = set()

    # 先选择包含多类的图像优先加入
    sorted_images = sorted(image_to_classes.items(), key=lambda x: -len(x[1]))
    for idx, classes in sorted_images:
        if covered_classes == set(range(1, 18)):
            break
        if not image_to_classes[idx].issubset(covered_classes):
            selected_indices.add(idx)
            covered_classes.update(image_to_classes[idx])

    # 补充其他图像直到满额
    remaining_indices = set(range(len(total_seg))) - selected_indices
    additional = list(remaining_indices)
    random.shuffle(additional)

    selected_indices.update(additional)
    selected_indices = list(selected_indices)

    # 划分 train / val
    random.shuffle(selected_indices)
    tv = len(selected_indices)
    tr = int(tv * train_percent)
    train_indices = set(selected_indices[:tr])
    val_indices = set(selected_indices[tr:])

    # 写入文件
    with open(os.path.join(saveBasePath, 'trainval.txt'), 'w') as ftrainval, \
        open(os.path.join(saveBasePath, 'train.txt'), 'w') as ftrain, \
        open(os.path.join(saveBasePath, 'val.txt'), 'w') as fval:

        for i in range(len(total_seg)):
            name = total_seg[i][:-4] + '\n'
            if i in train_indices or i in val_indices:
                ftrainval.write(name)
            if i in train_indices:
                ftrain.write(name)
            elif i in val_indices:
                fval.write(name)

    print(f"总共使用图像数量：{len(selected_indices)} / {len(total_seg)}")
    print(f"训练集：{len(train_indices)}，验证集：{len(val_indices)}")
    # ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    # ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    # ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    # fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    # for i in list:  
    #     name = total_seg[i][:-4]+'\n'  
    #     if i in trainval:  
    #         ftrainval.write(name)  
    #         if i in train:  
    #             ftrain.write(name)  
    #         else:  
    #             fval.write(name)  
    #     else:  
    #         ftest.write(name)  
    
    # ftrainval.close()  
    # ftrain.close()  
    # fval.close()  
    # ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums        = np.zeros([256], int)
    for i in tqdm(file_list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。"%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。"%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")