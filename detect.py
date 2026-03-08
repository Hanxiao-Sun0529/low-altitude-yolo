import torch
import torch.nn as nn
import os
import yaml
import datetime
from ultralytics import YOLO
from pathlib import Path
import traceback

# ==================== SCSA 小目标注意力模块 ====================
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
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
        self.small_target_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        ca_out = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        sa_out = x * sa

        enhance_mask = self.small_target_enhance(x)
        enhanced_out = x * enhance_mask

        output = (ca_out + sa_out + enhanced_out) / 3.0
        return output

# ==================== 小目标检测头增强 ====================
def add_small_target_enhancements(model):
    try:
        from ultralytics.nn.modules import Detect

        def enhance_detection_head(module):
            if isinstance(module, Detect):
                print("增强检测头以适应小目标检测...")
                if hasattr(module, 'anchors'):
                    module.anchors *= 0.7
                    print("缩小 anchor 尺寸以适应小目标")

        def recursive_enhance(module):
            for _, child in module.named_children():
                enhance_detection_head(child)
                recursive_enhance(child)

        recursive_enhance(model.model)
        return model
    except Exception as e:
        print(f"小目标增强失败: {e}")
        return model

# ==================== 工具函数 ====================
def check_environment():
    print("="*60)
    print("环境检查")
    print("="*60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)
    return True

def safe_yaml_load(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        return None

# ==================== 主训练流程 ====================
def main():
    if not check_environment():
        return

    yaml_path = r"C:\Users\shx\Desktop\配置教程\训练验证集\sunhx_dataset.yaml"
    if not os.path.exists(yaml_path):
        print(f"错误：未找到指定的 YAML 文件: {yaml_path}")
        return

    yaml_content = safe_yaml_load(yaml_path)
    if not yaml_content:
        print("YAML文件读取失败，请检查编码或内容！")
        return

    nc = yaml_content.get('nc', 0)
    class_names = yaml_content.get('names', {})

    print(f"数据集信息: {nc} 个类别")
    print("类别名称:", class_names)

    # 加载 YOLOv11n 模型
    print("\n加载 YOLOv11n 模型...")
    model = YOLO("yolo11n.pt")
    model.model.nc = nc

    # 添加小目标增强
    model = add_small_target_enhancements(model)

    # 训练配置
    training_config = {
        'data': yaml_path,
        'epochs': 250,
        'imgsz': 1024,
        'batch': 8,
        'workers': 8,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'optimizer': 'AdamW',
        'cos_lr': True,
        'close_mosaic': 15,
        'copy_paste': 0.5,
        'erasing': 0.2,
        'project': 'scsa_yolo11_training',
        'name': f'yolo11n_scsa_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}',
    }

    print("\n训练配置:")
    for k, v in training_config.items():
        if k not in ['data', 'project', 'name']:
            print(f"  {k}: {v}")

    print("\n开始训练...")
    try:
        results = model.train(**training_config)
        print("\n训练完成！")
    except Exception as e:
        print(f"训练失败: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
