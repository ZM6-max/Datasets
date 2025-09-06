import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import random
from utils.utils import cvtColor, preprocess_input
class ValDataset(Dataset):
    def __init__(self, input_shape, dataset_path):
        super(ValDataset, self).__init__()
        self.input_shape        = input_shape
        self.dataset_path       = dataset_path
        self.imgs = [img for img in os.listdir(dataset_path)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_name = self.imgs[index]
        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        img         = Image.open(os.path.join(self.dataset_path, image_name))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg  = self.get_random_data(img, self.input_shape)
        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float32)), [2,0,1])
        return jpg, image_name, cvtColor(img)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, input_shape):
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        iw, ih  = image.size
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        
        image       = image.resize((w,h), Image.BICUBIC)
        # new_image   = Image.new('RGB', [w, h], (128,128,128))
        # new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return image

class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float32)), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)
            # print(nw)
            # print(nh)
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)

        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


# DataLoader中collate_fn使用
# def Val_dataset_collate(batch):
#     images      = []
#     pngs        = [] #标签 数值为0-num_class
#     seg_labels  = [] #标签 数值为0或1 但有num_class个通道
#     for img, png, labels in batch:
#         images.append(img)
#         pngs.append(png)
#         seg_labels.append(labels)
#     images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     pngs        = torch.from_numpy(np.array(pngs)).long()
#     seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
#     return images, pngs, seg_labels

def Val_dataset_collate(batch):
    images      = []
    names = []
    orgi_imgs = []
    for img , name,orgi_img in batch:
        images.append(img)
        names.append(name)
        orgi_imgs.append(orgi_img)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    return images, names, orgi_imgs


import random
import torch
import numpy as np

def deeplab_dataset_collate(batch):

    images_list = []
    pngs_list = []
    seg_labels_list = []

    for item in batch:
        img, png, labels = item

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        if isinstance(png, np.ndarray):
            png = torch.from_numpy(png).long()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).float()

        images_list.append(img)
        pngs_list.append(png)
        seg_labels_list.append(labels)

    images = torch.stack(images_list)        # [B, 3, H, W]
    pngs = torch.stack(pngs_list)            # [B, H, W]
    seg_labels = torch.stack(seg_labels_list)  # [B, H, W, C]

    TARGET_CLASSES = [4, 11, 12, 14, 16]
    num_classes = seg_labels.shape[-1]

    if not TARGET_CLASSES or max(TARGET_CLASSES) >= num_classes:
        return images, pngs, seg_labels

    batch_size = images.shape[0]

    # 构建类别到样本索引映射
    class_to_indices = {cls: [] for cls in TARGET_CLASSES}
    for j in range(batch_size):
        for cls in TARGET_CLASSES:
            if torch.any(seg_labels[j, ..., cls] > 0.5):
                class_to_indices[cls].append(j)

    for i in range(batch_size):
        if random.random() > 0.4:
            continue

        # 随机选一个类，这个类在源图像中存在
        candidate_classes = [cls for cls in TARGET_CLASSES if class_to_indices[cls]]
        if not candidate_classes:
            continue

        cls = random.choice(candidate_classes)

        # 从该类对应的图像中选一个源图像，不能是自己
        candidate_indices = [j for j in class_to_indices[cls] if j != i]
        if not candidate_indices:
            continue

        j = random.choice(candidate_indices)

        source_mask = seg_labels[j, ..., cls] > 0.5
        if not torch.any(source_mask):
            continue  # double-checking, 防止mask为空

        mask_3ch = source_mask.unsqueeze(0).repeat(3, 1, 1)
        images[i] = torch.where(mask_3ch, images[j], images[i])

        pngs[i] = torch.where(
            source_mask,
            torch.full_like(pngs[i], cls),
            pngs[i]
        )

        new_label = torch.zeros_like(seg_labels[i])
        new_label[..., cls] = 1.0

        seg_labels[i] = torch.where(
            source_mask.unsqueeze(-1),
            new_label,
            seg_labels[i]
        )

    return images, pngs, seg_labels


