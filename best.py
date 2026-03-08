import os
import torch
import torch.nn as nn
import yaml
import cv2
import datetime
from ultralytics import YOLO
from pathlib import Path
from torchvision.ops import nms
import numpy as np

# ==================== 蛇形卷积模块 ====================
class SnakeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(SnakeConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = SnakeActivation(out_channels)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class SnakeActivation(nn.Module):
    def __init__(self, channels, alpha=1.0, trainable=True):
        super(SnakeActivation, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * alpha) if trainable else alpha

    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-8)) * torch.pow(torch.sin(self.alpha * x), 2)

# ==================== 小目标优化注意力模块 ====================
class SmallTargetSCSAAttention(nn.Module):
    def __init__(self, in_channels, reduction=8, kernel_size=3):
        super(SmallTargetSCSAAttention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1))
        return x * ca * sa

# ==================== P2 检测头 ====================
def try_add_p2_detection_head(model):
    try:
        from ultralytics.nn.modules import Detect
        detect_layer = None
        for m in model.model.modules():
            if isinstance(m, Detect):
                detect_layer = m
                break
        if detect_layer is None:
            print("未找到 Detect 层，跳过 P2 检测头添加")
            return model

        print("找到 Detect 层，尝试添加 P2 检测头...")
        in_channels_list = detect_layer.ch
        p2_channels = in_channels_list[0] // 2
        new_in_channels = [p2_channels] + in_channels_list
        detect_layer.ch = new_in_channels
        detect_layer.stride = [detect_layer.stride[0] / 2] + detect_layer.stride
        print(f"添加 P2 检测头成功，新的 stride: {detect_layer.stride}")
        return model
    except Exception as e:
        print(f"P2 检测头添加失败: {e}")
        return model

# ==================== 激活函数替换 ====================
def add_snake_activation_to_model(model):
    def replace_act(module):
        for name,  child in module.named_children():
            if isinstance(child, (nn.SiLU, nn.ReLU)):
                setattr(module, name, SnakeActivation(64))
            else:
                replace_act(child)
    replace_act(model.model)
    return model

# ==================== 创建优化模型 ====================
def create_small_target_optimized_model(nc, model_size='m'):
    model_map = {'n': 'yolo11n.pt','s': 'yolo11s.pt','m': 'yolo11m.pt','l': 'yolo11l.pt','x': 'yolo11x.pt'}
    model = YOLO(model_map.get(model_size, 'yolo11m.pt'))
    model.model.nc = nc
    print(f"✅ 模型加载成功, 类别数={nc}")
    return model

# ==================== 融合多尺度推理结果 ====================
def merge_detections(detections_list, iou_thres=0.6):
    if len(detections_list) == 0:
        return []
    all_boxes = torch.cat(detections_list, dim=0)
    keep = nms(all_boxes[:, :4], all_boxes[:, 4], iou_thres)
    return all_boxes[keep]

def multiscale_and_merge(model, source, conf=0.001, iou=0.7, img_sizes=[640, 960, 1280], save_dir="runs/merged_results"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    dataset = os.listdir(source)
    all_scale_results = []

    for size in img_sizes:
        print(f"🔍 推理分辨率: {size}, conf={conf}, iou={iou}")
        results = model.predict(source=source, imgsz=size, conf=conf, iou=iou, max_det=300, save=False, verbose=False)
        all_scale_results.append(results)

    for i, img_name in enumerate(dataset):
        img_path = os.path.join(source, img_name)
        img = cv2.imread(img_path)
        detections_list = []
        for scale_results in all_scale_results:
            if i >= len(scale_results): continue
            result = scale_results[i].boxes.xyxy.cpu()
            confs = scale_results[i].boxes.conf.cpu().unsqueeze(1)
            cls = scale_results[i].boxes.cls.cpu().unsqueeze(1)
            det = torch.cat([result, confs, cls], dim=1)
            detections_list.append(det)
        merged = merge_detections(detections_list, iou_thres=iou)

        # 可视化
        for *xyxy, conf_v, cls_v in merged:
            label = f"{int(cls_v)} {conf_v:.2f}"
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (xyxy[0], xyxy[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        save_path = Path(save_dir) / img_name
        cv2.imwrite(str(save_path), img)

    print(f"\n🎯 多尺度结果融合完成！保存路径: {save_dir}")

# ==================== Grid Search 自动搜索 ====================
def grid_search_nms(model, data_yaml, source, conf_list=None, iou_list=None):
    if conf_list is None: conf_list = [0.001, 0.01, 0.05, 0.1]
    if iou_list is None: iou_list = [0.5, 0.6, 0.7, 0.8]

    best_map50, best_conf, best_iou = -1, None, None
    for conf in conf_list:
        for iou in iou_list:
            metrics = model.val(data=data_yaml, imgsz=640, conf=conf, iou=iou, verbose=False)
            map50 = metrics.results_dict.get("metrics/mAP50(B)", 0)
            print(f"Grid Search: conf={conf}, iou={iou}, mAP50={map50:.4f}")
            if map50 > best_map50:
                best_map50, best_conf, best_iou = map50, conf, iou

    print(f"\n✅ 最优 NMS 参数: conf={best_conf}, iou={best_iou}, mAP50={best_map50:.4f}")
    return best_conf, best_iou

# ==================== 小目标优化训练 ====================
def enhanced_small_target_training(yaml_path, nc):
    model = create_small_target_optimized_model(nc, 'm')
    model = try_add_p2_detection_head(model)
    model = add_snake_activation_to_model(model)

    training_config = {
        'data': yaml_path,
        'epochs': 250,
        'imgsz': 1024,
        'batch': 4,
        'optimizer': 'AdamW',
        'lr0': 0.002,
        'cos_lr': True,
        'patience': 50,
        'augment': True,
        'copy_paste': 0.5,
        'erasing': 0.2,
        'box': 7.0,
        'cls': 0.7,
        'dfl': 1.0,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'project': 'exp_small_targets',
        'name': f"final_opt_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    }

    results = model.train(**training_config)
    return model, results

# ==================== 主函数 ====================
def main():
    yaml_path = r"C:\Users\NBUFE\Desktop\SHXYXZX\训练验证集\sunhx_dataset.yaml"
    source = r"C:\Users\NBUFE\Desktop\SHXYXZX\训练验证集\images\val"

    if not os.path.exists(yaml_path):
        print("❌ YAML 文件未找到")
        return
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
    nc = yaml_content.get('nc', 0)

    # 1. 训练
    model, results = enhanced_small_target_training(yaml_path, nc)
    print("✅ 训练完成")

    # 2. Grid Search 找最优 conf/iou
    best_conf, best_iou = grid_search_nms(model, yaml_path, source)

    # 3. 多尺度 + 融合推理 (使用最佳参数)
    multiscale_and_merge(model, source, conf=best_conf, iou=best_iou)

if __name__ == '__main__':
    main()