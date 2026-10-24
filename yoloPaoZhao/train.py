import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'E:\文档\伦羽文档\AI\AI\yoloPaoZhao\ultralytics\cfg\models\11\yolo11.yaml').load('yolo11n.pt')
    model.train(data=r'E:\文档\伦羽文档\AI\AI\yoloPaoZhao\ultralytics\cfg\datasets\coco128.yaml',
                imgsz=960,
                epochs=240,
                batch=16,
                workers=4,
                device=0,

                degrees=20,  # 图像旋转范围（角度）
                scale=0.15,  # 设置图像缩放因子，影响数据增强中的尺度变化。
                mosaic=1.0,  # 启用 Mosaic 数据增强，值为 1.0 表示完全启用。
                mixup=0.0,  # 设置 MixUp 数据增强的混合比例，这里设置为 0.0，表示不使用。
                copy_paste=0.3,  # 设置 Copy-Paste 数据增强的概率。

                exist_ok=True,
                optimizer='SGD',
                # close_mosaic=10,
                # resume=False,
                project='runs/train',
                name='yolo11n',
                single_cls=False,
                # cache=False,
                )
    results = model.val(data="coco128.yaml",imgsz=960)