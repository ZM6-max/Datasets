import os
import shutil
#------------复制Val图片--------------#
# 图片所在的文件夹路径
source_folder = 'VOCdevkit\VOC2007\JPEGImages'  # 请替换为你的图片所在文件夹的实际路径
# 包含图片名称的txt文档路径
image_names_file = 'VOCdevkit\VOC2007\ImageSets\Segmentation/val.txt'  # 请替换为你的txt文档的实际路径
# 新的文件夹路径，用于存储找到的图片
destination_folder = 'img'  # 请替换为你希望存储图片的新文件夹的实际路径
# 如果新文件夹不存在，则创建它
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
# 读取txt文档中的图片名称
with open(image_names_file, 'r') as file:
    image_names = file.read().splitlines()
# 遍历图片名称列表，查找并复制图片到新文件夹
for image_name in image_names:
    source_path = os.path.join(source_folder, f"{image_name}.jpg")
    if os.path.exists(source_path):
        destination_path = os.path.join(destination_folder, f"{image_name}.jpg")
        shutil.copy2(source_path, destination_path)  # 使用copy2保留元数据（如时间戳）
    else:
        print(f"图片 {image_name}.jpg 未在 {source_folder} 中找到。")
print("所有找到的图片已复制到新文件夹。")
#-------------------------------------#
