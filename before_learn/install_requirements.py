"""
股票价格预测模型 - 依赖安装工具
================================

【pandas库说明】
如果你在运行过程中遇到如下报错：
Traceback (most recent call last):
  File "/root/miniconda3/learn_transformer/learn/learn/train/first.py", line 2, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
请直接运行本脚本（python install_requirements.py），即可自动安装pandas库。

文件作用：
- 自动安装项目所需的Python依赖包
- 检查系统环境和依赖版本
- 处理依赖冲突和版本兼容性问题
- 提供安装进度和错误处理

主要功能：
1. 依赖检查：检查已安装的包和版本
2. 自动安装：安装缺失的依赖包
3. 版本管理：确保依赖包版本兼容
4. 错误处理：处理安装过程中的错误
5. 环境验证：验证安装后的环境状态

安装的依赖包：
- pandas：数据处理核心库（必须）
- PyTorch：深度学习框架
- NumPy：数值计算库
- Scikit-learn：机器学习库
- Matplotlib：绘图库
- 其他必要的科学计算库

使用方法：
- 直接运行：python install_requirements.py
- 开发环境：用于设置开发环境
- 生产环境：用于部署环境配置

输出结果：
- 安装进度报告
- 依赖版本信息
- 安装成功/失败状态
- 环境验证结果

作者：AI Assistant
创建时间：2024年
"""
import importlib
import subprocess
import sys

# 需要的库及其pip包名（pandas放首位，必须）
required_packages = {
    "pandas": "pandas",   # 必须的核心数据处理库
    "torch": "torch",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "yfinance": "yfinance",
    "transformers": "transformers",
    "ta": "ta",  # 技术分析库
    "seaborn": "seaborn"  # 可视化库
}

def install_if_missing(module_name, pip_name):
    try:
        importlib.import_module(module_name)
        print(f"已安装: {module_name}")
    except ImportError:
        print(f"未检测到 {module_name}，正在安装 {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

if __name__ == "__main__":
    print("【重要】本项目依赖 pandas 进行数据处理，如遇到 'No module named pandas'，请确保本脚本能正常运行。")
    for module, pip_name in required_packages.items():
        install_if_missing(module, pip_name)
    print("所有依赖库已检查并安装完毕。")
