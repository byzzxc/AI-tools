"""
数据集准备脚本：将原始 datasets/train 整理成 YOLO 标准格式
- 自动划分 train / val（默认 8:2）
- 复制图片和标签到正确目录
- 生成 data.yaml
用法：python prepare_dataset.py
"""
import os
import random
import shutil
from pathlib import Path

# ========= 配置 =========
WORKSPACE = Path(__file__).parent.resolve()
SRC_DIR = WORKSPACE / "datasets" / "train"
DST_DIR = WORKSPACE / "datasets"
VAL_RATIO = 0.2          # 验证集比例
SEED = 42                 # 随机种子，保证可复现

# 类别（从 classes - 副本.txt 读取的中文名称）
CLASSES = [
    "黑点",
    "网纹不良",
    "半粒",
    "空包",
    "颜色",
    "字符",
]


def read_label_safe(path):
    """读取标签文件，兼容 utf-8 / gbk / ansi"""
    for enc in ('utf-8', 'gbk', 'ansi'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    return ""


def main():
    random.seed(SEED)

    src_images = SRC_DIR / "images"
    src_labels = SRC_DIR / "labels"

    if not src_images.exists():
        print(f"❌ 图片目录不存在: {src_images}")
        return

    # 收集所有图片（支持 bmp / jpg / png），过滤掉不存在/损坏的
    exts = {'.bmp', '.jpg', '.jpeg', '.png'}
    all_images = []
    for f in os.listdir(src_images):
        p = Path(f)
        if p.suffix.lower() in exts:
            full_path = src_images / f
            if full_path.is_file():
                all_images.append(f)
    all_images.sort()  # 排序保证稳定

    print(f"📦 发现图片总数: {len(all_images)}")

    # 打乱并划分
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - VAL_RATIO))
    train_files = sorted(all_images[:split_idx])
    val_files = sorted(all_images[split_idx:])

    print(f"  训练集: {len(train_files)} 张")
    print(f"  验证集: {len(val_files)} 张")

    # 清空旧的目标目录（避免残留）
    for phase in ['train', 'val']:
        for sub in ['images', 'labels']:
            p = DST_DIR / phase / sub
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    # 复制函数
    skipped = []
    def copy_pair(img_name, phase):
        stem = Path(img_name).stem
        img_src = src_images / img_name
        lbl_src = src_labels / f"{stem}.txt"

        img_dst = DST_DIR / phase / "images" / img_name
        lbl_dst = DST_DIR / phase / "labels" / f"{stem}.txt"

        try:
            shutil.copy2(img_src, img_dst)
        except Exception as e:
            skipped.append(img_name)
            return
        # 标签文件不存在或为空时，创建空标签
        label_content = read_label_safe(lbl_src) if lbl_src.exists() else ""
        with open(lbl_dst, 'w', encoding='utf-8') as f:
            f.write(label_content)

    # 执行复制
    print("📋 复制训练集...")
    for f in train_files:
        copy_pair(f, 'train')

    print("📋 复制验证集...")
    for f in val_files:
        copy_pair(f, 'val')

    # ========= 生成 data.yaml =========
    yaml_path = DST_DIR / "data.yaml"
    yaml_content = f"""# YOLO 数据集配置 - 产品缺陷检测
path: {DST_DIR}  # 数据集根目录（绝对路径）
train: train/images
val: val/images

nc: {len(CLASSES)}  # 类别数量
names: {CLASSES}
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n✅ 完成！")
    if skipped:
        print(f"⚠️  跳过 {len(skipped)} 个损坏/不可读文件")
    print(f"📁 目录结构:")
    print(f"  datasets/")
    print(f"    ├── data.yaml          ← {yaml_path.name}")
    print(f"    ├── train/images/      ({len(train_files)} 张)")
    print(f"    ├── train/labels/")
    print(f"    ├── val/images/        ({len(val_files)} 张)")
    print(f"    └── val/labels/")
    print(f"\n🏷️  类别 ({len(CLASSES)}):")
    for i, name in enumerate(CLASSES):
        print(f"  {i}: {name}")
    print(f"\n💡 UI 中使用的路径: datasets/data.yaml")


if __name__ == '__main__':
    main()
