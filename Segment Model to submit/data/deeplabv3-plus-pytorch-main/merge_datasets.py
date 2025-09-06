import os
import shutil
import glob

def expand_dataset_with_prefix(A_dir, B_dir, jpeg_dir, seg_dir, train_txt, prefix="pse_"):
    
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    
   
    A_files = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(A_dir, '*'))}
    B_files = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(B_dir, '*'))}
    
    
    common_files = A_files & B_files
    print(f"找到 {len(common_files)} 个共同文件")
    
   
    existing_jpegs = {os.path.splitext(f)[0] for f in os.listdir(jpeg_dir)}
    existing_segs = {os.path.splitext(f)[0] for f in os.listdir(seg_dir)}
    existing_train = set()
    
 
    if os.path.exists(train_txt):
        with open(train_txt, 'r') as f:
            existing_train = set(line.strip() for line in f)
    
    new_entries = []
    
    
    for base_name in common_files:
       
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            path = os.path.join(A_dir, base_name + ext)
            if os.path.exists(path):
                img_path = path
                break
        
        if not img_path:
            print(f"警告: 在A目录中未找到 {base_name} 的图像文件")
            continue
     
        prefixed_name = prefix + base_name
        new_img_name = f"{prefixed_name}.jpg"
        
       
        if prefixed_name in existing_jpegs:
            print(f"跳过 {base_name} - 目标图像已存在")
            continue
            
       
        dest_img = os.path.join(jpeg_dir, new_img_name)
        
      
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                rgb_img = img.convert('RGB')
                rgb_img.save(dest_img, 'JPEG')
            print(f"转换并复制图像: {base_name} -> {prefixed_name}")
        except ImportError:
            
            shutil.copy(img_path, dest_img)
            print(f"警告: 非JPG文件 {base_name} 未转换 (安装Pillow可自动转换)")
       
        new_entries.append(prefixed_name)
    
   
    for base_name in common_files:
       
        label_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            path = os.path.join(B_dir, base_name + ext)
            if os.path.exists(path):
                label_path = path
                break
        
        if not label_path:
            print(f"警告: 在B目录中未找到 {base_name} 的标签文件")
            continue
        
        
        prefixed_name = prefix + base_name
        new_label_name = f"{prefixed_name}.png"
        
       
        if prefixed_name in existing_segs:
            print(f"跳过 {base_name} - 目标标签已存在")
            continue
            
       
        dest_label = os.path.join(seg_dir, new_label_name)
        
       
        try:
            from PIL import Image
            with Image.open(label_path) as img:
                img.save(dest_label, 'PNG')
            print(f"转换并复制标签: {base_name} -> {prefixed_name}")
        except ImportError:
           
            shutil.copy(label_path, dest_label)
            print(f"警告: 非PNG文件 {base_name} 未转换 (安装Pillow可自动转换)")
    
   
    added_count = 0
    with open(train_txt, 'a') as f:
        for entry in new_entries:
            if entry not in existing_train:
                f.write(f"{entry}\n")
                added_count += 1
                print(f"添加到训练集: {entry}")
    
    print("\n操作完成! 统计报告:")
    print(f"添加图像: {len(new_entries)} 个")
    print(f"添加训练集条目: {added_count} 个")
    print(f"JPEGImages 现在包含: {len(os.listdir(jpeg_dir))} 个文件")
    print(f"SegmentationClass 现在包含: {len(os.listdir(seg_dir))} 个文件")
    print(f"train.txt 现在包含: {len(existing_train) + added_count} 个条目")


if __name__ == "__main__":
    A_directory = "/data/tangchen/deeplabv3-plus-pytorch-main/Notag_TC"          # 原始图像目录
    B_directory = "/data/tangchen/deeplabv3-plus-pytorch-main/pseudo_labels"     # 伪标签目录
    jpeg_dir = "/data/tangchen/deeplabv3-plus-pytorch-main/New_DataSets/VOC2007/JPEGImages"    # 目标图像目录
    seg_dir = "/data/tangchen/deeplabv3-plus-pytorch-main/New_DataSets/VOC2007/SegmentationClass" # 目标分割标签目录
    train_file = "/data/tangchen/deeplabv3-plus-pytorch-main/New_DataSets/VOC2007/ImageSets/Segmentation/train.txt"    # 训练集列表文件
    
    expand_dataset_with_prefix(
        A_directory, 
        B_directory, 
        jpeg_dir, 
        seg_dir, 
        train_file,
        prefix="pse_"  # 简写pseudo_labels
    )