"""
YOLO11 可视化训练工具 - Streamlit UI
整合：图片导入 + LabelImg 标注 + 训练监控 + 历史查看
启动：streamlit run train_ui.py
"""
import streamlit as st
import subprocess
import os
import sys
import time
import json
import shutil
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
import glob

# ============== 页面配置 ==============
st.set_page_config(
    page_title="YOLO11 可视化训练工具",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

WORKSPACE = Path(__file__).parent.resolve()

# 自动搜索 LabelImg 目录（可移植）
def find_labelimg():
    """按优先级搜索 LabelImg 位置"""
    candidates = [
        WORKSPACE.parent / "biaozhugongju" / "labelImg",  # 同级 biaozhugongju
        WORKSPACE / "labelImg",                              # 项目内 labelImg
        Path(__file__).parent.parent.parent / "biaozhugongju" / "labelImg",
    ]
    for c in candidates:
        if (c / "labelImg.py").exists() and (c / "libs").exists():
            return c
    return None

LABELIMG_DIR = find_labelimg()

# 数据集目录约定
DATA_IMAGES = WORKSPACE / "datasets" / "train" / "images"
DATA_LABELS = WORKSPACE / "datasets" / "train" / "labels"

# 训练超参数持久化：浏览器刷新会清空 session_state，这里落盘到项目内，刷新后自动恢复
UI_PARAMS_FILE = WORKSPACE / ".yolo_config" / "train_params.json"


def load_train_params():
    try:
        with open(UI_PARAMS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_train_params(params):
    """参数有变化才写盘，避免训练时每 3 秒自动刷新都写文件"""
    try:
        if params == load_train_params():
            return
        UI_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(UI_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _pref_int(prefs, key, default, lo, hi):
    try:
        v = int(prefs.get(key, default))
    except Exception:
        v = default
    return min(max(v, lo), hi)


def _pref_float(prefs, key, default, lo, hi):
    try:
        v = float(prefs.get(key, default))
    except Exception:
        v = default
    return min(max(v, lo), hi)


def _seed(key, value):
    """只在首次渲染时把保存值塞进 session_state，之后由 Streamlit 自己维护"""
    if key not in st.session_state:
        st.session_state[key] = value


# ============== GPU 状态 ==============
def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            return f"✅ {name} ({mem_mb:.0f}MB)"
        else:
            return "⚠️ CPU 模式"
    except Exception:
        return "❓ 未知"


# ============== 侧边栏 ==============
def build_sidebar():
    # 上次保存的参数（浏览器刷新后 session_state 会丢失，这里从磁盘恢复）
    prefs = load_train_params()
    with st.sidebar:
        st.header("🎯 YOLO11 可视化训练工具")
        st.caption(get_gpu_info())

        st.divider()
        st.header("1. 模型配置")

        sizes = ["n", "s", "m", "l", "x"]
        size_idx = sizes.index(prefs['model_size']) if prefs.get('model_size') in sizes else 0
        model_size = st.selectbox("模型规模", sizes, index=size_idx, key="sb_model_size")
        model_weight = f"yolo11{model_size}.pt"

        resume = st.checkbox("🔁 加载已有 best.pt 续训", value=False)
        if resume:
            best_pts = list(WORKSPACE.glob("runs/train/*/weights/best.pt"))
            if best_pts:
                choice = st.selectbox("选择续训权重", [str(p.relative_to(WORKSPACE)) for p in best_pts], 0)
                model_weight = str(WORKSPACE / choice)
            else:
                st.warning("未找到已有 best.pt")

        resume_ckpt = st.checkbox("⏩ 断点续训（从 last.pt 继续，沿用原 run）", value=False)
        if resume_ckpt:
            last_pts = sorted(
                WORKSPACE.glob("runs/train/*/weights/last.pt"),
                key=lambda x: x.stat().st_mtime, reverse=True,
            )
            if last_pts:
                ck = st.selectbox("选择 last.pt", [str(p.relative_to(WORKSPACE)) for p in last_pts], 0)
                model_weight = str(WORKSPACE / ck)
            else:
                st.warning("未找到已有 last.pt")

        st.divider()
        st.header("2. 数据集")

        data_options = sorted(
            [str(p.relative_to(WORKSPACE)) for p in WORKSPACE.glob("ultralytics/cfg/datasets/*.yaml")]
        ) + sorted(
            [str(p.relative_to(WORKSPACE)) for p in WORKSPACE.glob("**/data.yaml")]
        )
        data_options = list(dict.fromkeys(data_options))

        # 默认选上次保存的数据集；没有记录则选自己的数据集（datasets/data.yaml），其次 coco128。
        # 注意 Windows 下选项是反斜杠路径，统一成正斜杠再匹配
        norm = [o.replace('\\', '/') for o in data_options]
        default_idx = 0
        saved_data = str(prefs.get('data_path') or '').replace('\\', '/')
        if saved_data in norm:
            default_idx = norm.index(saved_data)
        else:
            for pref in ("datasets/data.yaml", "ultralytics/cfg/datasets/coco128.yaml"):
                if pref in norm and (WORKSPACE / pref).exists():
                    default_idx = norm.index(pref)
                    break
        data_path = st.selectbox("数据集 YAML", data_options, index=default_idx, key="sb_data_yaml")

        custom_data = st.text_input("或自定义路径", value=prefs.get('custom_data', ''),
                                    placeholder="留空用上方", key="sb_custom_data")
        if custom_data:
            data_path = custom_data

        # 显示类别信息
        try:
            import re
            with open(WORKSPACE / data_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            nc_match = re.search(r'nc:\s*(\d+)', content)
            names_match = re.search(r'names:\s*\[([^\]]+)\]', content)
            if nc_match:
                st.caption(f"📊 nc={nc_match.group(1)} 类")
            if names_match:
                names = [n.strip().strip("'\"") for n in names_match.group(1).split(',')]
                st.caption(f"🏷️  {', '.join(names)}")
        except Exception:
            pass

        st.divider()
        st.header("3. 训练超参数")

        c1, c2 = st.columns(2)
        with c1:
            _seed("sb_epochs", _pref_int(prefs, 'epochs', 240, 1, 1000))
            epochs = st.number_input("Epochs", 1, 1000, step=1, key="sb_epochs")
            _seed("sb_batch", _pref_int(prefs, 'batch', 16, 1, 128))
            batch = st.number_input("Batch", 1, 128, step=1, key="sb_batch")
            _seed("sb_imgsz", _pref_int(prefs, 'imgsz', 640, 32, 2048))
            imgsz = st.number_input("Imgsz", 32, 2048, step=32, key="sb_imgsz")
        with c2:
            devices = ["0", "cpu", "1", "2"]
            dev_idx = devices.index(prefs['device']) if prefs.get('device') in devices else 0
            device = st.selectbox("Device", devices, index=dev_idx, key="sb_device")
            optimizers = ["SGD", "Adam", "AdamW", "auto"]
            opt_idx = optimizers.index(prefs['optimizer']) if prefs.get('optimizer') in optimizers else 0
            optimizer = st.selectbox("Optimizer", optimizers, index=opt_idx, key="sb_optimizer")
            _seed("sb_workers", _pref_int(prefs, 'workers', 1, 0, 16))
            workers = st.number_input("Workers", 0, 16, step=1, key="sb_workers")

        c3, c4 = st.columns(2)
        with c3:
            _seed("sb_lr0", _pref_float(prefs, 'lr0', 0.01, 0.0001, 1.0))
            lr0 = st.number_input("lr0", 0.0001, 1.0, step=0.0001, format="%.4f", key="sb_lr0")
        with c4:
            _seed("sb_lrf", _pref_float(prefs, 'lrf', 0.01, 0.001, 1.0))
            lrf = st.number_input("lrf", 0.001, 1.0, step=0.001, format="%.3f", key="sb_lrf")

        st.divider()
        st.header("4. 数据增强")
        _seed("sb_degrees", _pref_float(prefs, 'degrees', 180.0, 0.0, 180.0))
        deg = st.slider("旋转 degrees", 0.0, 180.0, step=1.0, key="sb_degrees")
        _seed("sb_scale", _pref_float(prefs, 'scale', 0.5, 0.0, 1.0))
        scale = st.slider("缩放 scale", 0.0, 1.0, step=0.05, key="sb_scale")
        _seed("sb_mosaic", _pref_float(prefs, 'mosaic', 1.0, 0.0, 1.0))
        mosaic = st.slider("Mosaic", 0.0, 1.0, step=0.05, key="sb_mosaic")
        _seed("sb_mixup", _pref_float(prefs, 'mixup', 0.0, 0.0, 1.0))
        mixup = st.slider("MixUp", 0.0, 1.0, step=0.05, key="sb_mixup")
        _seed("sb_copy_paste", _pref_float(prefs, 'copy_paste', 0.1, 0.0, 1.0))
        copy_paste = st.slider("Copy-Paste", 0.0, 1.0, step=0.05, key="sb_copy_paste")
        _seed("sb_close_mosaic", _pref_int(prefs, 'close_mosaic', 10, 0, 50))
        close_mosaic = st.number_input("最后N轮关Mosaic", 0, 50, step=1, key="sb_close_mosaic")

        st.divider()
        st.header("5. 输出")
        run_name_default = f"yolo11{model_size}_{datetime.now().strftime('%m%d_%H%M')}"
        run_name = st.text_input("Run 名称", value=run_name_default)
        project = st.text_input("Project", value=prefs.get('project') or "runs/train", key="sb_project")

    params = {
        'model_size': model_size,
        'data_path': data_path, 'custom_data': custom_data,
        'epochs': int(epochs), 'batch': int(batch), 'imgsz': int(imgsz),
        'device': device, 'optimizer': optimizer, 'workers': int(workers),
        'lr0': float(lr0), 'lrf': float(lrf),
        'degrees': float(deg), 'scale': float(scale), 'mosaic': float(mosaic),
        'mixup': float(mixup), 'copy_paste': float(copy_paste),
        'close_mosaic': int(close_mosaic),
        'project': project,
    }
    # 参数变化才落盘，训练中每 3 秒自动刷新不会反复写
    save_train_params(params)

    return {
        'model_weight': model_weight, 'data_path': data_path,
        'epochs': epochs, 'batch': batch, 'imgsz': imgsz,
        'device': device, 'optimizer': optimizer, 'workers': workers,
        'lr0': lr0, 'lrf': lrf,
        'degrees': deg, 'scale': scale, 'mosaic': mosaic,
        'mixup': mixup, 'copy_paste': copy_paste, 'close_mosaic': close_mosaic,
        'run_name': run_name, 'project': project, 'resume': resume,
        'resume_ckpt': resume_ckpt,
    }


# ============== 构建训练命令 ==============
def build_cmd(p):
    cmd = [sys.executable, str(WORKSPACE / "train_tool.py")]
    def add(n, v):
        if isinstance(v, bool):
            if v: cmd.append(f"--{n}")
        elif v not in (None, ''):
            cmd.extend([f"--{n}", str(v)])
    add('model', p['model_weight'])
    if p.get('resume_ckpt'):
        cmd.append('--resume')
    add('data', p['data_path'])
    add('epochs', p['epochs'])
    add('batch', p['batch'])
    add('imgsz', p['imgsz'])
    add('device', p['device'])
    add('workers', p['workers'])
    add('optimizer', p['optimizer'])
    add('lr0', p['lr0'])
    add('lrf', p['lrf'])
    add('degrees', p['degrees'])
    add('scale', p['scale'])
    add('mosaic', p['mosaic'])
    add('mixup', p['mixup'])
    add('copy_paste', p['copy_paste'])
    add('close_mosaic', p['close_mosaic'])
    add('name', p['run_name'])
    add('project', p['project'])
    return cmd


def read_csv(path):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


def parse_per_class_metrics(run_dir):
    """从 train.log 解析最后一次验证输出的逐类别指标表（类别 / P / R / mAP50 / mAP50-95）"""
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None

    # 找最后一次验证的表头行（每个 epoch 验证都会打一次，取最后一次）
    header_idx = -1
    for i, line in enumerate(lines):
        if ("Class" in line and "Images" in line
                and "Instances" in line and "mAP50" in line):
            header_idx = i
    if header_idx < 0:
        return None

    rows = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        # 格式: 类别名  Images  Instances  P  R  mAP50  mAP50-95
        m = re.match(
            r'^(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$',
            stripped,
        )
        if m:
            rows.append({
                "类别": m.group(1),
                "Images": int(m.group(2)),
                "Instances": int(m.group(3)),
                "P": float(m.group(4)),
                "R": float(m.group(5)),
                "mAP50": float(m.group(6)),
                "mAP50-95": float(m.group(7)),
            })
        elif rows:
            break
    return pd.DataFrame(rows) if rows else None


def latest_run_dir(project):
    """按输出文件的最新修改时间取最近的训练目录"""
    base = WORKSPACE / project
    if not base.exists():
        return None
    dirs = [d for d in base.iterdir()
            if d.is_dir() and ((d / "args.yaml").exists() or (d / "results.csv").exists())]
    if not dirs:
        return None

    def key(d):
        times = [d.stat().st_mtime]
        for f in ("results.csv", "train.log", "args.yaml"):
            p = d / f
            if p.exists():
                times.append(p.stat().st_mtime)
        return max(times)

    return max(dirs, key=key)


def resolve_run_dir(project, default_name):
    """优先用本会话启动的 run；页面刷新导致会话丢失时回退到最新 run"""
    base = WORKSPACE / project
    name = st.session_state.get('run_name')
    if name and (base / name).exists():
        return base / name
    latest = latest_run_dir(project)
    return latest if latest else base / default_name


def train_process_running():
    """是否有 train_tool.py 进程在跑（刷新页面后也能恢复训练状态）"""
    try:
        import psutil
        for p in psutil.process_iter(['cmdline']):
            cmdline = p.info.get('cmdline') or []
            if any('train_tool.py' in str(a) for a in cmdline):
                return True
    except Exception:
        pass
    return False


def find_training_processes():
    """找出所有在跑 train_tool.py 的进程（含 UI 之外启动的，如计划任务/命令行）"""
    found = []
    try:
        import psutil
        for pr in psutil.process_iter(['pid', 'cmdline']):
            cmdline = pr.info.get('cmdline') or []
            if any('train_tool.py' in str(a) for a in cmdline):
                found.append(pr)
    except Exception:
        pass
    return found


def stop_training_processes():
    """杀掉所有训练进程及其子进程（DataLoader worker），返回 (已停止PID, 剩余数量)"""
    stopped = []
    for pr in find_training_processes():
        try:
            children = pr.children(recursive=True)
        except Exception:
            children = []
        for cp in children:
            try:
                cp.kill()
            except Exception:
                pass
        try:
            pr.kill()
            stopped.append(pr.pid)
        except Exception:
            pass
        try:
            pr.wait(timeout=5)
        except Exception:
            pass
    return stopped, len(find_training_processes())


# ============== Tab: 📤 导入图片 ==============
def tab_import_images():
    st.subheader("📤 批量导入图片")
    st.caption(f"目标目录: `{DATA_IMAGES}`")

    DATA_IMAGES.mkdir(parents=True, exist_ok=True)
    DATA_LABELS.mkdir(parents=True, exist_ok=True)

    uploaded = st.file_uploader(
        "上传图片（支持多选）",
        type=['bmp', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="支持 bmp / jpg / jpeg / png",
    )

    if uploaded:
        # 防重复：每次 rerun（如拖动滑块）uploader 都会返回全部已上传文件，
        # 同名同大小的视为已保存过，直接跳过
        new_files = []
        skipped = 0
        for uf in uploaded:
            dst = DATA_IMAGES / uf.name
            if dst.exists() and dst.stat().st_size == uf.size:
                skipped += 1
            else:
                new_files.append(uf)

        if new_files:
            progress = st.progress(0, text=f"正在保存 0/{len(new_files)} 张...")
            for i, uf in enumerate(new_files):
                safe_name = uf.name
                dst = DATA_IMAGES / safe_name
                # 如果重名加后缀
                n = 1
                while dst.exists():
                    stem = Path(safe_name).stem
                    ext = Path(safe_name).suffix
                    dst = DATA_IMAGES / f"{stem}_{n}{ext}"
                    n += 1
                with open(dst, 'wb') as f:
                    f.write(uf.read())
                progress.progress((i + 1) / len(new_files), text=f"保存中 {i+1}/{len(new_files)}")
            msg = f"✅ 成功保存 {len(new_files)} 张图片！"
            if skipped:
                msg += f"（跳过已存在 {skipped} 张）"
            st.success(msg)

    # 统计
    exts = {'.bmp', '.jpg', '.jpeg', '.png'}
    all_imgs = sorted([f for f in os.listdir(DATA_IMAGES)
                       if Path(f).suffix.lower() in exts
                       and (DATA_IMAGES / f).is_file()])
    all_labels = [Path(f).stem + '.txt' for f in all_imgs]

    has_label = 0
    empty_label = 0
    no_label = 0
    for stem in [Path(f).stem for f in all_imgs]:
        lp = DATA_LABELS / f"{stem}.txt"
        if lp.exists():
            content = open(lp, 'r', encoding='utf-8', errors='ignore').read().strip()
            if content:
                has_label += 1
            else:
                empty_label += 1
        else:
            no_label += 1

    # 进度卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📷 图片总数", len(all_imgs))
    c2.metric("✅ 已标注", has_label)
    c3.metric("📝 空标签(负样本)", empty_label)
    c4.metric("❌ 未标注", no_label, delta_color="inverse")

    # 一键清空（危险操作，二次确认）
    with st.expander("🗑️ 清空数据集（删除全部图片和标签）"):
        st.warning(f"⚠️ 将永久删除 train/val 下的全部图片和标签（当前共 {len(all_imgs)} 张图片），不可恢复！")
        confirm_del = st.checkbox("我确认要删除全部图片和标签", key="confirm_del_all")
        if st.button("🗑️ 立即删除全部", disabled=not confirm_del, key="btn_del_all"):
            del_img, del_lbl = 0, 0
            for phase in ['train', 'val']:
                img_dir = WORKSPACE / "datasets" / phase / "images"
                lbl_dir = WORKSPACE / "datasets" / phase / "labels"
                if img_dir.exists():
                    for f in list(img_dir.iterdir()):
                        if f.is_file() and f.suffix.lower() in exts:
                            f.unlink()
                            del_img += 1
                if lbl_dir.exists():
                    for f in list(lbl_dir.iterdir()):
                        if f.is_file() and f.suffix.lower() == '.txt':
                            f.unlink()
                            del_lbl += 1
            # 清理缓存文件
            for cache in WORKSPACE.glob("datasets/**/*.cache"):
                cache.unlink()
            st.success(f"✅ 已删除 {del_img} 张图片、{del_lbl} 个标签文件")
            time.sleep(1)
            st.rerun()

    # 缩略图网格
    st.divider()
    st.subheader("🖼 数据集预览")
    if all_imgs:
        show_n = st.slider("显示数量", min(len(all_imgs), 6), len(all_imgs),
                           min(10, len(all_imgs)), key="img_show_n")
        cols = st.columns(5)
        for i, fname in enumerate(all_imgs[:show_n]):
            with cols[i % 5]:
                img_path = DATA_IMAGES / fname
                # 状态徽章
                stem = Path(fname).stem
                lp = DATA_LABELS / f"{stem}.txt"
                if lp.exists():
                    content = open(lp, 'r', encoding='utf-8', errors='ignore').read().strip()
                    badge = "✅" if content else "⚪"
                else:
                    badge = "❌"
                st.image(str(img_path), caption=f"{badge} {fname}", use_column_width=True)
        if len(all_imgs) > show_n:
            st.caption(f"... 还有 {len(all_imgs) - show_n} 张未显示")
    else:
        st.info("暂无图片，请上方上传 👆")

    # 一键生成 data.yaml
    st.divider()
    st.subheader("📋 生成 data.yaml")
    col_classes, col_names = st.columns([1, 3])
    with col_classes:
        nc = st.number_input("类别数", 1, 100, 14, key="import_nc")
    with col_names:
        default_names = "空泡,无药,热封不良,泡眼不良,头发,异物,剪切不良,半粒,黑点,PVC异物,药片异物,PVC接头,批号异常,药片不良"
        names_str = st.text_input("类别名称 (逗号分隔，顺序须与 LabelImg predefined_classes.txt 一致)", default_names, key="import_names")
    names = [n.strip() for n in names_str.split(',') if n.strip()]

    if st.button("⚡ 一键划分 train/val 并生成 data.yaml", use_container_width=True, key="btn_gen_yaml"):
        import random
        random.seed(42)
        # 用 prepare_dataset.py 里同样的逻辑
        exts2 = {'.bmp', '.jpg', '.jpeg', '.png'}
        imgs = [f for f in os.listdir(DATA_IMAGES)
                if Path(f).suffix.lower() in exts2
                and (DATA_IMAGES / f).is_file()]
        random.shuffle(imgs)
        split_idx = int(len(imgs) * 0.8)
        train_imgs = sorted(imgs[:split_idx])
        val_imgs = sorted(imgs[split_idx:])

        for phase, file_list in [('train', train_imgs), ('val', val_imgs)]:
            for sub in ['images', 'labels']:
                p = WORKSPACE / "datasets" / phase / sub
                p.mkdir(parents=True, exist_ok=True)
            for fname in file_list:
                stem = Path(fname).stem
                src_img = DATA_IMAGES / fname
                src_lbl = DATA_LABELS / f"{stem}.txt"
                dst_img = WORKSPACE / "datasets" / phase / "images" / fname
                dst_lbl = WORKSPACE / "datasets" / phase / "labels" / f"{stem}.txt"
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
                # 总是同步最新标签（划分后继续标注，重新划分可刷新 val 标签）；
                # 注意 train 阶段 src==dst，跳过自我复制
                if src_lbl.exists() and src_lbl.resolve() != dst_lbl.resolve():
                    shutil.copy2(src_lbl, dst_lbl)
                elif not dst_lbl.exists():
                    dst_lbl.touch()

        yaml_path = WORKSPACE / "datasets" / "data.yaml"
        yaml_content = f"""# YOLO 数据集配置
path: {WORKSPACE / 'datasets'}
train: train/images
val: val/images

nc: {len(names)}
names: {names}
"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        st.success(f"✅ 完成！train={len(train_imgs)}, val={len(val_imgs)}, data.yaml 已生成")
        st.code(str(yaml_path))


# ============== Tab: ✏️ 标注图片 ==============
def _labelimg_python():
    """选 LabelImg 用的 Python 解释器。
    Anaconda conda Qt 的 qjpeg 插件损坏（jpg 无法解码），
    优先用 python.org Python38（已装 pip PyQt5，JPEG 正常）。"""
    candidates = [
        r"C:\Program Files\Python38\python.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return sys.executable


def tab_label_images():
    st.subheader("✏️ 标注图片（LabelImg）")

    DATA_IMAGES.mkdir(parents=True, exist_ok=True)
    DATA_LABELS.mkdir(parents=True, exist_ok=True)

    # ===== 一键启动 LabelImg =====
    global LABELIMG_DIR
    if LABELIMG_DIR is None:
        st.warning("⚠️ 未自动找到 LabelImg，请指定路径")
        manual_path = st.text_input("LabelImg 目录 (含 labelImg.py)", value="",
                                    placeholder="例如: D:/tools/labelImg")
        if manual_path and Path(manual_path).exists():
            LABELIMG_DIR = Path(manual_path)

    if LABELIMG_DIR is not None and (LABELIMG_DIR / "labelImg.py").exists():
        labelimg_py = LABELIMG_DIR / "labelImg.py"
        st.markdown("### 🚀 一键启动 LabelImg")
        st.caption(f"位置: {labelimg_py} | 解释器: {_labelimg_python()}")
        col_start, col_fmt = st.columns([3, 1])
        with col_start:
            if st.button("🎨 打开 LabelImg 标注工具", type='primary', use_container_width=True, key='btn_open_labelimg'):
                env = os.environ.copy()
                env['PYTHONPATH'] = str(LABELIMG_DIR) + os.pathsep + env.get('PYTHONPATH', '')
                # LabelImg 参数顺序: image_dir class_file save_dir
                class_file = LABELIMG_DIR / "data" / "predefined_classes.txt"
                cmd = [_labelimg_python(), str(labelimg_py), str(DATA_IMAGES), str(class_file), str(DATA_LABELS)]
                try:
                    subprocess.Popen(cmd, cwd=str(LABELIMG_DIR), env=env)
                    st.success("✅ LabelImg 已启动！图片和标签目录已自动加载。")
                    st.info("💡 快捷键: W=新建框, D=下一张, A=上一张, Ctrl+S=保存, Ctrl+Y=切YOLO格式")
                except Exception as e:
                    st.error(f"启动失败: {e}")
        with col_fmt:
            st.markdown("**标注格式**")
            st.code("YOLO\nclass x_center y_center w h")
        st.divider()

    # 统计（即使没图片也显示 0）
    exts = {'.bmp', '.jpg', '.jpeg', '.png'}
    all_imgs = sorted([f for f in os.listdir(DATA_IMAGES)
                       if Path(f).suffix.lower() in exts
                       and (DATA_IMAGES / f).is_file()])

    c1, c2, c3 = st.columns(3)
    c1.metric("📷 图片数", len(all_imgs))
    c2.metric("📂 图片目录", f"datasets/train/images")
    c3.metric("📂 标签目录", f"datasets/train/labels")

    if len(all_imgs) == 0:
        st.warning("⚠️ datasets/train/images 里还没有图片。请先到「📤 导入图片」Tab 上传，或直接用 LabelImg 的「打开目录」功能选择图片文件夹。")
        return

    # ===== 标注进度 =====
    st.markdown("### 📊 标注进度")
    has_label = 0
    empty_label = 0
    no_label = 0
    detail = []
    for fname in all_imgs:
        stem = Path(fname).stem
        lp = DATA_LABELS / f"{stem}.txt"
        if lp.exists():
            content = open(lp, 'r', encoding='utf-8', errors='ignore').read().strip()
            if content:
                has_label += 1
                # 统计每个文件有几个框
                n_boxes = len([l for l in content.split('\n') if l.strip()])
                detail.append((fname, "✅", n_boxes))
            else:
                empty_label += 1
                detail.append((fname, "⚪", 0))
        else:
            no_label += 1
            detail.append((fname, "❌", 0))

    total = len(all_imgs)
    done = has_label + empty_label
    pct = done / total * 100 if total else 0

    st.progress(pct / 100, text=f"标注进度: {done}/{total} ({pct:.1f}%)")

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("📷 图片总数", total)
    dc2.metric("✅ 已标注", has_label)
    dc3.metric("⚪ 空标签", empty_label)
    dc4.metric("❌ 未标注", no_label, delta_color="inverse")

    # 未标注列表
    if no_label > 0:
        st.markdown("**❌ 未标注图片（前20张）:**")
        not_done = [d for d in detail if d[1] == "❌"][:20]
        st.code('\n'.join([f[0] for f in not_done]))

    # 带标注文件的框数统计
    if has_label > 0:
        st.markdown("**📦 标注文件框数统计:**")
        annotated = [(f, n) for f, s, n in detail if n > 0]
        if annotated:
            df_boxes = pd.DataFrame(annotated, columns=['文件', '框数'])
            c5, c6, c7 = st.columns(3)
            c5.metric("总框数", int(df_boxes['框数'].sum()))
            c6.metric("平均/张", f"{df_boxes['框数'].mean():.1f}")
            c7.metric("最多/张", int(df_boxes['框数'].max()))

    st.divider()

    # ===== 查看单张标注 =====
    st.markdown("### 🔍 查看单张标注")
    pick = st.selectbox("选择图片", all_imgs, index=0 if all_imgs else None, key="pick_img")
    if pick:
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.image(str(DATA_IMAGES / pick), caption=pick, use_column_width=True)
        with pcol2:
            stem = Path(pick).stem
            lp = DATA_LABELS / f"{stem}.txt"
            st.markdown(f"**标签文件:** `{stem}.txt`")
            if lp.exists():
                content = open(lp, 'r', encoding='utf-8', errors='ignore').read()
                st.code(content if content.strip() else "（空标签 - 负样本）", language=None)
            else:
                st.warning("❌ 还没有标签文件")


# ============== 主面板（训练相关 Tab） ==============
def build_training_tabs(p):
    tab_live, tab_history, tab_config, tab_log = st.tabs([
        "📈 实时训练", "📚 历史训练", "⚙️ 当前配置", "📜 日志"
    ])

    # ---- 实时训练 ----
    with tab_live:
        run_dir = WORKSPACE / p['project'] / st.session_state.get('run_name', p['run_name'])
        csv_path = run_dir / "results.csv"
        png_path = run_dir / "results.png"

        if csv_path.exists():
            df = read_csv(str(csv_path))
            if df is not None and len(df) > 0:
                cur_ep = len(df)
                total = p['epochs']
                prog = min(cur_ep / total, 1.0)
                st.progress(prog, text=f"Epoch: {cur_ep}/{total}  ({prog*100:.1f}%)")

                last = df.iloc[-1]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("box_loss", f"{last.get('train/box_loss', 0):.4f}")
                m2.metric("mAP50", f"{last.get('metrics/mAP50(B)', 0):.4f}")
                m3.metric("mAP50-95", f"{last.get('metrics/mAP50-95(B)', 0):.4f}")
                m4.metric("recall", f"{last.get('metrics/recall(B)', 0):.4f}")

                loss_cols = [c for c in df.columns if 'loss' in c.lower() and 'val' not in c]
                if loss_cols:
                    st.markdown("**训练损失**")
                    st.line_chart(df[loss_cols].copy(), height=220)

                val_loss = [c for c in df.columns if 'loss' in c.lower() and 'val' in c]
                if val_loss:
                    st.markdown("**验证损失**")
                    st.line_chart(df[val_loss].copy(), height=220)

                map_cols = [c for c in df.columns if 'map' in c.lower()]
                if map_cols:
                    st.markdown("**mAP 指标**")
                    st.line_chart(df[map_cols].copy(), height=220)
            else:
                st.info("⏳ 等待数据...")
        else:
            st.info("ℹ️ 点击训练后实时显示")

        if png_path.exists():
            st.divider()
            st.markdown("**综合结果图**")
            st.image(str(png_path), use_column_width=True)

    # ---- 历史训练 ----
    with tab_history:
        st.subheader("📚 历史训练记录")
        runs = sorted(WORKSPACE.glob("runs/train/*/results.csv"))
        if not runs:
            st.info("暂无历史训练")
        else:
            selected = st.selectbox(
                "选择训练",
                [str(r.parent.relative_to(WORKSPACE)) for r in runs],
            )
            sel_dir = WORKSPACE / selected
            sel_csv = sel_dir / "results.csv"

            args_path = sel_dir / "args.yaml"
            if args_path.exists():
                with open(args_path, 'r', encoding='utf-8', errors='ignore') as f:
                    args = f.read()
                with st.expander("训练配置", expanded=False):
                    st.code(args, language='yaml')

            df = read_csv(str(sel_csv))
            if df is not None and len(df) > 0:
                st.metric("总 Epochs", len(df))
                last = df.iloc[-1]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("最终 mAP50-95", f"{last.get('metrics/mAP50-95(B)', 0):.4f}")
                c2.metric("最佳 mAP50-95", f"{df['metrics/mAP50-95(B)'].max():.4f}" if 'metrics/mAP50-95(B)' in df.columns else "-")
                c3.metric("最终 mAP50", f"{last.get('metrics/mAP50(B)', 0):.4f}")

                st.markdown("**完整曲线**")
                chart_cols = [c for c in df.columns if any(k in c for k in ['loss', 'mAP', 'precision', 'recall', 'lr'])]
                if chart_cols:
                    st.line_chart(df[chart_cols].copy(), height=400)

            png = sel_dir / "results.png"
            if png.exists():
                st.markdown("**综合结果图**")
                st.image(str(png), use_column_width=True)

            weights = sel_dir / "weights"
            if weights.exists():
                st.markdown("**权重文件**")
                for wf in sorted(weights.iterdir()):
                    sz = wf.stat().st_size / (1024**2)
                    st.caption(f"📎 {wf.name}  ({sz:.1f} MB)")

    # ---- 当前配置 ----
    with tab_config:
        st.subheader("⚙️ 本次训练配置")
        cfg = {
            'model': p['model_weight'], 'data': p['data_path'],
            'epochs': p['epochs'], 'batch': p['batch'], 'imgsz': p['imgsz'],
            'device': p['device'], 'optimizer': p['optimizer'],
            'workers': p['workers'], 'lr0': p['lr0'], 'lrf': p['lrf'],
            'degrees': p['degrees'], 'scale': p['scale'], 'mosaic': p['mosaic'],
            'mixup': p['mixup'], 'copy_paste': p['copy_paste'],
            'close_mosaic': p['close_mosaic'],
            'project': p['project'], 'name': p['run_name'],
        }
        st.json(cfg, expanded=True)
        st.markdown("**完整命令:**")
        st.code(' '.join(build_cmd(p)))

    # ---- 日志 ----
    with tab_log:
        st.subheader("📜 训练日志")
        run_dir = WORKSPACE / p['project'] / st.session_state.get('run_name', p['run_name'])
        log_path = run_dir / "train.log"
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            st.code(''.join(lines[-500:]), language=None)
        else:
            st.info("训练启动后显示日志")


# ============== 主入口 ==============
def main():
    p = build_sidebar()

    # 监控目录：本会话启动的 run；页面刷新导致会话丢失时回退到最新一次训练
    run_dir = resolve_run_dir(p['project'], p['run_name'])
    # 训练是否在进行：会话内进程活着，或系统中存在 train_tool.py 进程
    training_live = bool(st.session_state.get('is_training', False)) or train_process_running()

    # 标题 + 控制按钮
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.title("🎯 YOLO11 可视化训练工具")
    with c2:
        st.metric("状态", "🟢 训练中" if training_live else "⚪ 就绪")
    with c3:
        st.metric("GPU", get_gpu_info().split()[0] if "CPU" not in get_gpu_info() else "CPU")

    st.divider()

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        go = st.button("🚀 开始训练", disabled=training_live, type='primary', use_container_width=True)
    with bc2:
        stop = st.button("⏹ 停止训练", disabled=not training_live, use_container_width=True)
    with bc3:
        st.button("🔄 刷新", use_container_width=True)

    if go:
        cmd = build_cmd(p)
        rn = p['run_name']
        if p.get('resume_ckpt'):
            # 断点续训会沿用 last.pt 所在的 run 目录，日志也写到同一目录
            rn = Path(p['model_weight']).parent.parent.name
        log_dir = WORKSPACE / p['project'] / rn
        log_dir.mkdir(parents=True, exist_ok=True)
        log_f = open(log_dir / "train.log", "w", encoding="utf-8")
        st.session_state['proc'] = subprocess.Popen(
            cmd, cwd=str(WORKSPACE), stdout=log_f, stderr=subprocess.STDOUT, text=True,
        )
        st.session_state['is_training'] = True
        st.session_state['run_name'] = rn
        st.session_state['project'] = p['project']
        st.success(f"✅ 已启动 PID={st.session_state['proc'].pid} | Run: {rn}")
        time.sleep(1)
        st.rerun()

    if stop:
        # 训练可能是 UI 启动的，也可能是计划任务/命令行启动的（刷新后会话丢失，proc 句柄不在），
        # 统一按进程名清理，避免只能靠命令行杀进程
        stopped, remaining = stop_training_processes()
        proc = st.session_state.get('proc')
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except Exception:
                pass
            st.session_state['proc'] = None
        st.session_state['is_training'] = False
        training_live = False  # 本轮的「训练中」状态立即失效，避免自动刷新循环

        if remaining > 0:
            st.error(f"❌ 仍有 {remaining} 个训练进程未退出，请再点一次「停止训练」")
        elif stopped:
            st.warning(f"⏹ 训练已停止（PID: {', '.join(map(str, stopped))}）")
            st.info("已保存的 last.pt 可直接用于断点续训；当前这一轮未完成的进度会丢失。")
        else:
            st.info("没有检测到正在运行的训练进程")

    if st.session_state.get('is_training', False) and st.session_state.get('proc', None):
        ret = st.session_state['proc'].poll()
        if ret is not None:
            st.session_state['is_training'] = False
            if ret == 0: st.success("✅ 训练完成！")
            else: st.error(f"❌ 异常退出 code={ret}")

    st.divider()

    # ======= 6 个 Tab =======
    tab_import, tab_label, tab_live, tab_history, tab_config, tab_log = st.tabs([
        "📤 导入图片", "✏️ 标注图片", "📈 实时训练", "📚 历史训练", "⚙️ 配置", "📜 日志"
    ])

    with tab_import:
        tab_import_images()

    with tab_label:
        tab_label_images()

    with tab_live:
        st.caption(f"监控目录: `{run_dir.relative_to(WORKSPACE)}`")
        csv_path = run_dir / "results.csv"
        png_path = run_dir / "results.png"
        batch_imgs = sorted(run_dir.glob("train_batch*.jpg"))
        if batch_imgs:
            st.markdown("**训练批次预览**")
            st.image(str(batch_imgs[0]), use_column_width=True)
        if csv_path.exists():
            df = read_csv(str(csv_path))
            if df is not None and len(df) > 0:
                cur_ep = len(df)
                total = p['epochs']
                prog = min(cur_ep / total, 1.0)
                st.progress(prog, text=f"Epoch: {cur_ep}/{total} ({prog*100:.1f}%)")
                last = df.iloc[-1]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("box_loss", f"{last.get('train/box_loss', 0):.4f}")
                m2.metric("mAP50", f"{last.get('metrics/mAP50(B)', 0):.4f}")
                m3.metric("mAP50-95", f"{last.get('metrics/mAP50-95(B)', 0):.4f}")
                m4.metric("recall", f"{last.get('metrics/recall(B)', 0):.4f}")
                loss_cols = [c for c in df.columns if 'loss' in c.lower() and 'val' not in c]
                if loss_cols:
                    st.markdown("**训练损失**")
                    st.line_chart(df[loss_cols].copy(), height=220)
                val_loss = [c for c in df.columns if 'loss' in c.lower() and 'val' in c]
                if val_loss:
                    st.markdown("**验证损失**")
                    st.line_chart(df[val_loss].copy(), height=220)
                map_cols = [c for c in df.columns if 'map' in c.lower()]
                if map_cols:
                    st.markdown("**mAP**")
                    st.line_chart(df[map_cols].copy(), height=220)
            else:
                st.info("⏳ 等待数据...")
        else:
            st.info("ℹ️ 点「🚀 开始训练」后实时显示")
        if png_path.exists():
            st.divider()
            st.image(str(png_path), use_column_width=True)

        # 逐类别最终指标
        per_class = parse_per_class_metrics(run_dir)
        if per_class is not None and len(per_class) > 0:
            st.divider()
            st.markdown("**🏷️ 逐类别最终指标（取最后一次验证）**")
            st.dataframe(per_class, use_container_width=True, hide_index=True)

    with tab_history:
        st.subheader("📚 历史训练")
        runs = sorted(WORKSPACE.glob("runs/train/*/results.csv"),
                      key=lambda x: x.stat().st_mtime, reverse=True)
        if not runs:
            st.info("暂无")
        else:
            selected = st.selectbox("选择", [str(r.parent.relative_to(WORKSPACE)) for r in runs], key='hist_select')
            sel_dir = WORKSPACE / selected
            df = read_csv(str(sel_dir / "results.csv"))
            if df is not None and len(df) > 0:
                st.metric("Epochs", len(df))
                last = df.iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("最终 mAP50-95", f"{last.get('metrics/mAP50-95(B)', 0):.4f}")
                c2.metric("最佳 mAP50-95", f"{df['metrics/mAP50-95(B)'].max():.4f}" if 'metrics/mAP50-95(B)' in df.columns else "-")
                c3.metric("最终 mAP50", f"{last.get('metrics/mAP50(B)', 0):.4f}")
                chart_cols = [c for c in df.columns if any(k in c for k in ['loss', 'mAP', 'precision', 'recall', 'lr'])]
                if chart_cols:
                    st.line_chart(df[chart_cols].copy(), height=400)
            png = sel_dir / "results.png"
            if png.exists():
                st.image(str(png), use_column_width=True)

            # 逐类别最终指标
            per_class = parse_per_class_metrics(sel_dir)
            if per_class is not None and len(per_class) > 0:
                st.markdown("**🏷️ 逐类别最终指标（取最后一次验证）**")
                st.dataframe(per_class, use_container_width=True, hide_index=True)
            elif not (sel_dir / "train.log").exists():
                st.info("ℹ️ 该 run 无 train.log，无法解析逐类别指标。")

            weights = sel_dir / "weights"
            if weights.exists():
                st.subheader("💾 模型文件")
                for wf in sorted(weights.iterdir()):
                    sz = wf.stat().st_size / (1024**2)
                    st.markdown(f"- `{wf.name}` ({sz:.1f} MB)")

                best_pt = weights / "best.pt"
                col_exp, col_open = st.columns(2)
                with col_exp:
                    export_clicked = st.button(
                        "📤 导出 ONNX（best.pt）",
                        disabled=not best_pt.exists(),
                        use_container_width=True,
                        key=f"onnx_exp_{selected}",
                    )
                with col_open:
                    open_clicked = st.button(
                        "📂 打开模型目录",
                        use_container_width=True,
                        key=f"open_dir_{selected}",
                    )

                if export_clicked:
                    with st.spinner("正在导出 ONNX，请稍候..."):
                        try:
                            from ultralytics import YOLO
                            model = YOLO(str(best_pt))
                            exp_imgsz = p['imgsz']
                            args_yaml = sel_dir / "args.yaml"
                            if args_yaml.exists():
                                try:
                                    import yaml
                                    with open(args_yaml, 'r', encoding='utf-8') as ay:
                                        ad = yaml.safe_load(ay) or {}
                                    if ad.get('imgsz'):
                                        exp_imgsz = int(ad['imgsz'])
                                except Exception:
                                    pass
                            onnx_path = model.export(format="onnx", imgsz=exp_imgsz)
                            st.success(f"✅ 导出成功：{onnx_path}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 导出失败：{e}")

                if open_clicked:
                    try:
                        os.startfile(str(weights))
                        st.info(f"已打开目录：{weights}")
                    except Exception as e:
                        st.error(f"无法打开目录：{e}")

                # 列出已生成的 ONNX 等可部署文件，提供下载
                deploy_files = []
                for ext in ("*.onnx", "*.engine", "*.torchscript", "*.xml", "*.bin"):
                    deploy_files.extend(weights.glob(ext))
                if deploy_files:
                    st.markdown("**📦 可部署文件**")
                    for dp in sorted(deploy_files):
                        dsz = dp.stat().st_size / (1024**2)
                        dc1, dc2 = st.columns([4, 1])
                        dc1.markdown(f"`{dp.name}` ({dsz:.1f} MB)")
                        try:
                            with open(dp, "rb") as fh:
                                dc2.download_button(
                                    "⬇️ 下载",
                                    data=fh.read(),
                                    file_name=dp.name,
                                    key=f"dl_{dp.name}_{selected}",
                                )
                        except Exception:
                            pass

    with tab_config:
        st.subheader("⚙️ 训练配置")
        cfg = {
            'model': p['model_weight'], 'data': p['data_path'],
            'epochs': p['epochs'], 'batch': p['batch'], 'imgsz': p['imgsz'],
            'device': p['device'], 'optimizer': p['optimizer'],
            'workers': p['workers'], 'lr0': p['lr0'], 'lrf': p['lrf'],
            'degrees': p['degrees'], 'scale': p['scale'], 'mosaic': p['mosaic'],
            'mixup': p['mixup'], 'copy_paste': p['copy_paste'],
            'close_mosaic': p['close_mosaic'],
            'project': p['project'], 'name': p['run_name'],
        }
        st.json(cfg, expanded=True)
        st.code(' '.join(build_cmd(p)))

    with tab_log:
        st.subheader("📜 日志")
        log_path = run_dir / "train.log"
        if log_path.exists():
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                st.code(''.join(f.readlines()[-500:]), language=None)
        else:
            st.info("训练启动后显示")

    # 自动刷新：训练进行中就持续刷新（刷新页面后同样生效）
    if training_live:
        time.sleep(3)
        st.rerun()

    st.divider()
    st.caption("YOLO11 可视化训练工具 · LabelImg 标注 · Streamlit")


if __name__ == "__main__":
    main()
