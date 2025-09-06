import os
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
color_to_label = \
{
    (0, 0, 0): 0,
    (127, 127, 0): 1,
    (0, 180, 33): 2,
    (127, 0, 0): 3,

}


def convert_mask_to_labels(mask_image):
    mask_array = np.array(mask_image)
    label_array = np.zeros(mask_array.shape[:2], dtype=np.int64)

    for color, label in color_to_label.items():
        color_array = np.array(color).reshape(1, 1, 3)
        label_array[np.all(mask_array == color_array, axis=-1)] = label

    return label_array


def process_image_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 处理常见的图像格式
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取图像
            mask_image = Image.open(input_path).convert('RGB')

            # 转换为标签图像
            # label_image_array = convert_mask_to_labels(mask_image)

            # 保存标签图像
            label_image = Image.fromarray(label_image_array.astype(np.uint8))
            label_image.save(output_path)
            print(f"Processed {filename} and saved to {output_path}")
# 处理单个图像的函数
def process_single_image(input_path, output_path):
    try:
        mask_image = Image.open(input_path).convert('RGB')
        label_image_array = convert_mask_to_labels(mask_image)
        label_image = Image.fromarray(label_image_array.astype(np.uint8))
        label_image.save(output_path)
        print(f"处理完成 {input_path}，保存到 {output_path}")
    except Exception as e:
        print(f"处理图像 {input_path} 时出错: {e}")

# 多线程处理文件夹内的图像
def process_image_folder_recursively_multithreaded(input_folder, output_folder, max_workers=8):
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(input_folder):
            relative_path = os.path.relpath(root, input_folder)
            current_output_folder = os.path.join(output_folder, relative_path)

            if not os.path.exists(current_output_folder):
                os.makedirs(current_output_folder)

            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_path = os.path.join(root, filename)
                    output_path = os.path.join(current_output_folder, filename)

                    # 提交任务到线程池
                    tasks.append(executor.submit(process_single_image, input_path, output_path))

        # 等待所有任务完成
        for task in tasks:
            task.result()

# 设置输入和输出文件夹路径
input_folder = r"VOCdevkit\VOC2007\anying转换格式"#你自标注彩图图像路径
output_folder = r"VOCdevkit\VOC2007\SegmentationClass"#训练路径

# 执行多线程处理
process_image_folder_recursively_multithreaded(input_folder, output_folder, max_workers=8)



