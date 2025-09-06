import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes', num_classes)  
    # 创建一个全是0的矩阵，是一个混淆矩阵
    hist = np.zeros((num_classes, num_classes))
    
    # 获得验证集标签路径列表和预测结果路径列表
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    # 读取每一个（图片-标签）对
    for ind in range(len(gt_imgs)): 
        # 读取预测结果
        pred = np.array(Image.open(pred_imgs[ind]))  
        # 读取对应的标签
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，跳过
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # 计算hist矩阵并累加
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  
        
        # 每计算10张输出一次mIoU值（忽略背景类）
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            # 计算忽略背景类的mIoU
            IoUs = per_class_iu(hist)
            non_bg_IoUs = IoUs[1:]  # 去除背景类（索引0）
            mIoU = np.nanmean(non_bg_IoUs) if len(non_bg_IoUs) > 0 else 0
            
            print('{:d} / {:d}: mIou-{:0.2f}% (ignoring background)'.format(
                ind, len(gt_imgs), 100 * mIoU))

    # 计算所有验证集图片的逐类别mIoU值
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    
    # 去除背景类的指标
    non_bg_IoUs = IoUs[1:]
    non_bg_PA_Recall = PA_Recall[1:]
    non_bg_Precision = Precision[1:]
    non_bg_name_classes = name_classes[1:] if name_classes is not None else None
    
    # 逐类别输出mIoU值（忽略背景类）
    if non_bg_name_classes is not None:
        for ind_class in range(len(non_bg_name_classes)):
            print('===>' + non_bg_name_classes[ind_class] + ':\tIou-' + 
                  str(round(non_bg_IoUs[ind_class] * 100, 2)) + 
                  '; Recall-' + str(round(non_bg_PA_Recall[ind_class] * 100, 2)) + 
                  '; Precision-' + str(round(non_bg_Precision[ind_class] * 100, 2)))

    # 计算忽略背景类的平均指标
    mIoU = np.nanmean(non_bg_IoUs) if len(non_bg_IoUs) > 0 else 0
    mPA = np.nanmean(non_bg_PA_Recall) if len(non_bg_PA_Recall) > 0 else 0
    
    # 计算忽略背景类的准确率
    # 正确预测的像素数（忽略背景类）
    correct_pixels = np.sum(hist[1:, 1:].diagonal())
    # 总像素数（忽略背景类）
    total_pixels = np.sum(hist[1:, :])
    accuracy = correct_pixels / (total_pixels + 1e-8)
    
    print('===> mIoU (ignoring background): ' + str(round(mIoU * 100, 2)) + 
          '; mPA: ' + str(round(mPA * 100, 2)) + 
          '; Accuracy: ' + str(round(accuracy * 100, 2)))  
    
    # 返回结果（包括完整的混淆矩阵和忽略背景类的指标）
    return hist, non_bg_IoUs, non_bg_PA_Recall, non_bg_Precision

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes[1:], "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes[1:], "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes[1:], "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes[1:], "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            