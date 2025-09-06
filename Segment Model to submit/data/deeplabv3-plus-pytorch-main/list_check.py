import os
train_txt_path = os.path.join('VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', 'train.txt')
val_txt_path = os.path.join('VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', 'val.txt')

# 读取文件名列表（去除换行符）
with open(train_txt_path, 'r') as f:
    train_list = set(line.strip() for line in f.readlines())

with open(val_txt_path, 'r') as f:
    val_list = set(line.strip() for line in f.readlines())

# 查找重复项
overlap = train_list & val_list

# 输出结果
if overlap:
    print(f"⚠️ 发现 {len(overlap)} 个重复的文件名存在于 train.txt 和 val.txt 中：")
    for name in sorted(overlap):
        print(name)
else:
    print("✅ train.txt 和 val.txt 没有重复的文件名。")