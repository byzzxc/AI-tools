@echo off
chcp 65001 >nul
title YOLO11 可视化训练工具

REM 使用 Anaconda Python（已装好 torch CUDA / streamlit / ultralytics / PyQt5）
set PY=D:\AI\Anaconda3\python.exe

echo ============================================
echo   YOLO11 可视化训练工具 启动中...
echo ============================================
echo.

REM 检查 Python
if not exist "%PY%" (
    echo [错误] 未找到 Anaconda Python: %PY%
    echo 请修改本脚本中的 PY 变量为你的 Python 路径
    pause
    exit /b 1
)

REM 检查依赖
"%PY%" -c "import streamlit, torch, ultralytics; print('[OK] 核心依赖已就绪')" >nul 2>&1
if errorlevel 1 (
    echo [安装] 首次运行，正在安装依赖...
    "%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [警告] 依赖安装可能不完整，请手动执行: pip install -r requirements.txt
    )
)

echo.
echo 启动 Streamlit UI...
echo 浏览器将自动打开 http://localhost:8501
echo 关闭本窗口即可停止服务
echo.

REM 启动
"%PY%" -m streamlit run train_ui.py --server.port 8501 --server.headless true
pause
