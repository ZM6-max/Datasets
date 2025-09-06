import os
import shutil

# 配置路径
val_txt_path = '/data/tangchen/deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt'          # 包含图片文件名的文本文件
source_folder = '/data/tangchen/deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/JPEGImages'    # 原始大文件夹路径
target_folder = 'val_img'             # 目标文件夹路径

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 读取val.txt文件
with open(val_txt_path, 'r') as f:
    image_names = [line.strip() for line in f.readlines()]

# 拷贝文件
for image_name in image_names:
    source_path = os.path.join(source_folder, image_name + ".jpg")
    target_path = os.path.join(target_folder, image_name + ".jpg")
    
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
        print(f'Copied: {image_name}')
    else:
        print(f'File not found: {image_name}')

print('Copy process completed!')