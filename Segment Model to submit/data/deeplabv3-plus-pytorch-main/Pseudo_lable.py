
import time
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.dataloader import ValDataset, Val_dataset_collate
from nets.deeplabv3_plus import DeepLab
import time
import numpy as np
import torch
import os
import torch.nn.functional as F
from PIL import Image
import cv2
from tqdm import tqdm

# # 配置参数
# PSEUDO_LABEL_SAVE_PATH = "/data/tangchen/deeplabv3-plus-pytorch-main/pseudo_labels"
# os.makedirs(PSEUDO_LABEL_SAVE_PATH, exist_ok=True)
# model_path1 =  "/data/tangchen/deeplabv3-plus-pytorch-main/logs/ep325-loss0.439-val_loss0.268_70.16.pth"
# model_path2 =  "/data/tangchen/deeplabv3-plus-pytorch-main/logs/ep250-loss0.444-val_loss0.259_69.62.pth"
# # 模型输入尺寸
# INPUT_SIZE = (512, 896)  # (width, height)
# # 原始图像尺寸
# ORIGINAL_SIZE = (2688, 1512)  # (width, height)
# # 滑动窗口参数
# # STRIDE = (448, 256)  # (width, height) - 50% 重叠
# # 置信度阈值
# CONF_THRESHOLD = 0.85

# # 加载模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model1 = DeepLab(num_classes=17, backbone='Caformer', downsample_factor=8, pretrained=False,decoder_dim=256).to(device)
# model1.load_state_dict(torch.load(model_path1, map_location=device))
# model1.eval()

# model2 = DeepLab(num_classes=17, backbone='Caformer', downsample_factor=8, pretrained=False,decoder_dim=256).to(device)
# model2.load_state_dict(torch.load(model_path2, map_location=device))
# model2.eval()
# val_dataset = ValDataset(input_shape=[512, 896], dataset_path='/data/tangchen/deeplabv3-plus-pytorch-main/Notag_TC/')
# # 创建数据加载器 (保持原有Dataset不变)
# val_dataLoader = DataLoader(
#     val_dataset, 
#     shuffle=False, 
#     batch_size=4,  # 批大小设为1，因为每张图很大
#     num_workers=4, 
#     pin_memory=True, 
#     drop_last=False, 
#     collate_fn=Val_dataset_collate
# )

# def generate_pseudo_labels():
#     print("开始生成伪标签...")
#     start_time = time.time()
    
#     for batch_idx, (images, filenames, orig_img)in enumerate(tqdm(val_dataLoader, desc="处理图像")):
#         # 假设Dataset返回图像和文件名

        
#         # # 获取原始尺寸信息
#         # if len(batch) >= 3:
#         #     original_sizes = batch[2]  # 假设Dataset返回原始尺寸
#         # else:
#         #     # 如果没有提供原始尺寸，使用默认值
#         #     original_sizes = [(2688, 1512)] * len(images)
#         # print(images)
#         images = images.to(device)
        
#         with torch.no_grad():
#             # 1. 双模型预测
#             # print(images.shape)
#             outputs1 = model1(images)
#             outputs2 = model2(images)
            
#             # 2. 转换为概率
#             probs1 = F.softmax(outputs1, dim=1)
#             probs2 = F.softmax(outputs2, dim=1)
            
#             # 3. 获取预测结果
#             preds1 = torch.argmax(probs1, dim=1)
#             preds2 = torch.argmax(probs2, dim=1)
            
#             # 4. 获取最大概率
#             max_probs1 = torch.max(probs1, dim=1)[0]
#             max_probs2 = torch.max(probs2, dim=1)[0]
            
#             # 5. 创建伪标签 (初始化为忽略值255)
#             pseudo_labels = torch.ones_like(preds1) * 17
            
#             # 6. 创建一致性掩码
#             consistency_mask = (preds1 == preds2)
#             high_conf_mask = (max_probs1 > CONF_THRESHOLD) & (max_probs2 > CONF_THRESHOLD)
#             valid_mask = consistency_mask & high_conf_mask
            
#             # 7. 应用筛选条件
#             pseudo_labels[valid_mask] = preds1[valid_mask]
            
#             # 8. 转换为CPU numpy数组
#             pseudo_labels = pseudo_labels.cpu().numpy().astype(np.uint8)
            
#             # 9. 保存伪标签
#             for i in range(len(images)):
#                 # 获取原始尺寸
#                 orig_w = np.array(orig_img[i]).shape[1]
#                 orig_h = np.array(orig_img[i]).shape[0]
#                 # 将预测结果resize回原始尺寸
#                 colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
#                             (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
#                             (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
#                             (128, 64, 12)]
#                 resized_label = cv2.resize(
#                     pseudo_labels[i], 
#                     (orig_w, orig_h), 
#                     interpolation=cv2.INTER_NEAREST  # 最近邻插值保持标签完整性
#                 )
#                 seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(resized_label, [-1])], [orig_h, orig_w, -1])
#                 # print(seg_img.shape)
#                 # print(np.array(orig_img[i]).shape)
#             #------------------------------------------------#
#             #   将新图片转换成Image的形式
#             #------------------------------------------------#
#                 seg_img   = Image.fromarray(np.uint8(seg_img))
#                 # new_image   = Image.blend(orig_img[i], seg_img, 0.6)
#                 # 保存伪标签
#                 save_path = os.path.join(PSEUDO_LABEL_SAVE_PATH, 
#                                         os.path.splitext(filenames[i])[0] + '.png')
#                 seg_img.save(save_path)
#                 # 可选：保存置信度图用于调试
#                 # conf_map = (max_probs1[i].cpu().numpy() * 255).astype(np.uint8)
#                 # resized_conf = cv2.resize(
#                 #     conf_map, 
#                 #     (orig_w, orig_h), 
#                 #     interpolation=cv2.INTER_NEAREST
#                 # )
#                 # conf_save_path = os.path.join(PSEUDO_LABEL_SAVE_PATH, 
#                 #                             os.path.splitext(filenames[i])[0] + '_confidence.png')
#                 # Image.fromarray(resized_conf).save(conf_save_path)
    
#     total_time = time.time() - start_time
#     print(f"伪标签生成完成! 处理 {len(val_dataLoader.dataset)} 张图像, 耗时: {total_time:.2f}秒")
#     print(f"伪标签已保存至: {PSEUDO_LABEL_SAVE_PATH}")

# # 执行生成
# generate_pseudo_labels()
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import heapq
from PIL import Image
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataloader import ValDataset, Val_dataset_collate
from nets.deeplabv3_plus import DeepLab

# 配置参数
PSEUDO_LABEL_SAVE_PATH = "/data/tangchen/deeplabv3-plus-pytorch-main/pseudo_labels"
os.makedirs(PSEUDO_LABEL_SAVE_PATH, exist_ok=True)
model_path1 =  "/data/tangchen/deeplabv3-plus-pytorch-main/logs/ep325-loss0.439-val_loss0.268_70.16.pth"
model_path2 =  "/data/tangchen/deeplabv3-plus-pytorch-main/logs/ep250-loss0.444-val_loss0.259_69.62.pth"
CONF_THRESHOLD = 0.85  # 置信度阈值

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1 = DeepLab(num_classes=17, backbone='Caformer', downsample_factor=8, pretrained=False, decoder_dim=256).to(device)
model1.load_state_dict(torch.load(model_path1, map_location=device))
model1.eval()

model2 = DeepLab(num_classes=17, backbone='Caformer', downsample_factor=8, pretrained=False, decoder_dim=256).to(device)
model2.load_state_dict(torch.load(model_path2, map_location=device))
model2.eval()

val_dataset = ValDataset(input_shape=[512, 896], dataset_path='/data/tangchen/deeplabv3-plus-pytorch-main/Notag_TC/')
val_dataLoader = DataLoader(
    val_dataset, 
    shuffle=False, 
    batch_size=8, 
    num_workers=4, 
    pin_memory=True, 
    drop_last=False, 
    collate_fn=Val_dataset_collate
)

def calculate_class_miou(pred1, pred2, num_classes=17):
    """
    计算每个类别的IoU并取平均得到mIoU
    
    参数:
        pred1: 模型1的预测 (B, H, W)
        pred2: 模型2的预测 (B, H, W)
        num_classes: 类别数量
    
    返回:
        每张图像的mIoU值 (B,)
    """
    batch_size = pred1.shape[0]
    miou_values = np.zeros(batch_size)
    
    for b in range(batch_size):
        class_ious = []
        for c in range(num_classes):
            # 计算交集
            intersection = ((pred1[b] == c) & (pred2[b] == c)).sum().float()
            
            # 计算并集
            union = ((pred1[b] == c) | (pred2[b] == c)).sum().float()
            
            # 避免除以零
            iou = intersection / (union + 1e-8)
            class_ious.append(iou.item())
        
        # 计算平均IoU (mIoU)
        miou_values[b] = np.mean(class_ious)
    
    return miou_values

def generate_pseudo_labels():
    print("开始生成伪标签...")
    start_time = time.time()
    
    # 使用最小堆保存前550个样本
    top_images = []  # 最小堆，大小为550
    
    # 颜色映射表
    colors = [ 
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), 
        (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
        (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
        (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
        (0, 64, 128), (128, 64, 12)
    ]
    
    # 单次遍历：计算mIoU并生成伪标签
    for batch_idx, (images, filenames, orig_img) in enumerate(tqdm(val_dataLoader, desc="处理图像")):
        images = images.to(device)
        
        with torch.no_grad():
            # 双模型推理
            outputs1 = model1(images)
            outputs2 = model2(images)
            
            # 获取预测结果
            preds1 = torch.argmax(outputs1, dim=1)
            preds2 = torch.argmax(outputs2, dim=1)
            
            # 获取模型2的概率图
            probs2 = F.softmax(outputs2, dim=1)
            max_probs2 = torch.max(probs2, dim=1)[0]  # 每个像素的最大概率
            
            # 计算每个类别的mIoU
            miou_values = calculate_class_miou(preds1, preds2)
            
            # 处理每张图像
            for i in range(len(images)):
                # 获取原始尺寸
                orig_img_i = orig_img[i]
                orig_w, orig_h = orig_img_i.size
                
                # 使用第二个模型生成伪标签基础
                base_pseudo_label = preds2[i].cpu().numpy().astype(np.uint8)
                
                # 应用置信度筛选策略
                max_prob_map = max_probs2[i].cpu().numpy()
                pseudo_label = np.where(max_prob_map >= CONF_THRESHOLD, base_pseudo_label, 17)
                
                # 调整到原始尺寸
                resized_label = cv2.resize(
                    pseudo_label, 
                    (orig_w, orig_h), 
                    interpolation=cv2.INTER_NEAREST
                )
                
                # 转换为彩色图像
                seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(resized_label, [-1])], 
                                     [orig_h, orig_w, -1])
                seg_img = Image.fromarray(np.uint8(seg_img))
                
                # 当前图像的mIoU
                current_miou = miou_values[i]
                
                # 使用堆维护前550个样本
                if len(top_images) < 550:
                    heapq.heappush(top_images, (current_miou, filenames[i], seg_img))
                else:
                    # 如果当前mIoU大于堆中最小的mIoU，则替换
                    if current_miou > top_images[0][0]:
                        heapq.heapreplace(top_images, (current_miou, filenames[i], seg_img))
    
    # 保存选中的伪标签
    print(f"正在保存 {len(top_images)} 张伪标签...")
    for miou, filename, seg_img in top_images:
        save_path = os.path.join(PSEUDO_LABEL_SAVE_PATH, 
                                os.path.splitext(filename)[0] + '.png')
        seg_img.save(save_path)
    
    total_time = time.time() - start_time
    min_miou = top_images[0][0] if top_images else 0
    max_miou = max([item[0] for item in top_images]) if top_images else 0
    print(f"伪标签生成完成! 共生成 {len(top_images)} 张伪标签, 耗时: {total_time:.2f}秒")
    print(f"mIoU范围: {min_miou:.4f} - {max_miou:.4f}")
    print(f"伪标签已保存至: {PSEUDO_LABEL_SAVE_PATH}")

# 执行生成
generate_pseudo_labels()