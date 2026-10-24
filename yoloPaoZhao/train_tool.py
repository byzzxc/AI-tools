"""
YOLO11n 训练子进程脚本
由 Streamlit UI 通过 subprocess 调用，避免 UI 进程阻塞
"""
import argparse
import sys
import json
import os
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='YOLO11 Training Tool')
    # 基础配置
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='模型权重或配置文件路径')
    parser.add_argument('--data', type=str, required=True, help='数据集配置 YAML 路径')
    parser.add_argument('--cfg', type=str, default=None, help='模型 YAML 配置路径（用于从头训练）')
    # 训练超参
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0', help='设备编号 (0/1/cpu)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW', 'auto'])
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率因子')
    # 数据增强
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--copy_paste', type=float, default=0.0)
    # 输出配置
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='yolo11n')
    parser.add_argument('--single_cls', action='store_true', default=False)
    parser.add_argument('--close_mosaic', type=int, default=0)
    parser.add_argument('--exist_ok', action='store_true', default=True)
    parser.add_argument('--resume', action='store_true', default=False,
                        help='从 last.pt 断点续训（沿用该 run 原有参数和目录）')

    args = parser.parse_args()

    # 写一份 run 信息到 JSON，让 UI 可以找到训练输出目录
    run_info = {
        'project': args.project,
        'name': args.name,
        'started': True,
    }
    if not args.resume:
        os.makedirs(args.project, exist_ok=True)
        with open(os.path.join(args.project, f'{args.name}_run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)

    # 设备处理：'cpu' -> cpu，'0'/'1' -> int
    device = args.device
    if device == 'cpu':
        device_arg = 'cpu'
    else:
        try:
            device_arg = int(device)
        except ValueError:
            device_arg = device

    # 构造训练 kwargs
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device_arg,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        degrees=args.degrees,
        scale=args.scale,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        project=args.project,
        name=args.name,
        single_cls=args.single_cls,
        close_mosaic=args.close_mosaic,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"  YOLO11 Training Starting")
    print(f"  Model: {args.model}")
    print(f"  Data:  {args.data}")
    print(f"  Output: {args.project}/{args.name}")
    print(f"{'='*60}\n", flush=True)

    # ultralytics 默认把 settings/缓存/字体写到 %APPDATA%\Ultralytics，
    # 本机该目录不可写（中文字体 Arial.Unicode.ttf 下载会失败导致训练直接退出），
    # 重定向到项目内的可写目录。必须在 import ultralytics 之前设置。
    if not os.environ.get('YOLO_CONFIG_DIR'):
        cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.yolo_config')
        os.makedirs(cfg_dir, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = cfg_dir

    # 类别名是中文，matplotlib 默认 Arial 缺中文字形（结果图里显示方块），
    # 指定系统自带中文字体
    try:
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    from ultralytics import YOLO

    if args.resume:
        # 断点续训：从 last.pt 继续，ultralytics 会自动读回该 run 的原始参数
        print(f"  Resume: 从 {args.model} 继续训练\n", flush=True)
        model = YOLO(args.model)
        results = model.train(resume=True)
    else:
        # 支持两种模式：加载预训练权重 / 加载 cfg 从头训练
        if args.cfg:
            model = YOLO(args.cfg).load(args.model)
        else:
            model = YOLO(args.model)
        results = model.train(**train_kwargs)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best model saved at: {args.project}/{args.name}/weights/best.pt")
    print(f"{'='*60}\n", flush=True)
