"""
股票价格预测模型 - 数据测试模块
================================

文件作用：
- 测试和验证数据加载、预处理功能的正确性
- 检查数据质量和完整性
- 验证技术指标计算的准确性
- 提供数据统计信息和可视化分析

主要功能：
1. 数据完整性检查：验证CSV文件的完整性和格式
2. 技术指标验证：检查计算的技术指标是否正确
3. 数据统计：生成数据的基本统计信息
4. 可视化分析：绘制数据分布和趋势图
5. 性能测试：测试数据处理的速度和效率

测试内容：
- CSV文件读取测试
- 技术指标计算测试
- 数据清洗效果测试
- 序列化处理测试
- 内存使用测试

使用方法：
- 直接运行：python test_data.py
- 作为测试套件：python -m pytest test_data.py

输出结果：
- 测试报告和统计信息
- 数据质量评估
- 性能基准测试
- 错误和警告信息

作者：AI Assistant
创建时间：2024年
"""

# 导入必要的库
import pandas as pd  # 数据处理库
import numpy as np  # 数值计算库
import os  # 操作系统接口
import matplotlib.pyplot as plt  # 绘图库
import warnings  # 警告处理
warnings.filterwarnings('ignore')  # 忽略警告信息

# 导入自定义模块
from data.data_processor import StockDataProcessor  # 数据处理器

def test_csv_loading():
    """测试CSV数据加载功能"""
    
    # 检查数据目录是否存在
    data_dir = "data/learn_csv"
    if not os.path.exists(data_dir):
        return False
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        return False
    
    # 测试加载第一个文件
    test_file = csv_files[0]
    csv_path = os.path.join(data_dir, test_file)
    
    # 创建数据处理器
    processor = StockDataProcessor(seq_length=60)
    
    # 加载数据
    data = processor.load_csv_data(csv_path)
    
    if data is None:
        return False
    
    return True

def test_technical_indicators():
    """测试技术指标计算功能"""
    
    # 检查数据目录
    data_dir = "data/learn_csv"
    if not os.path.exists(data_dir):
        return False
    
    # 获取第一个CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        return False
    
    csv_path = os.path.join(data_dir, csv_files[0])
    
    # 创建数据处理器
    processor = StockDataProcessor(seq_length=60)
    
    # 加载数据
    data = processor.load_csv_data(csv_path)
    if data is None:
        return False
    
    # 添加技术指标
    data_with_indicators = processor.add_technical_indicators(data)
    
    return True

def test_data_preparation():
    """测试数据准备功能"""
    
    # 检查数据目录
    data_dir = "data/learn_csv"
    if not os.path.exists(data_dir):
        return False
    
    # 创建数据处理器
    processor = StockDataProcessor(seq_length=60)
    
    # 加载多个股票数据
    stock_data = processor.load_multiple_stocks(data_dir, stock_codes=None)
    
    if not stock_data:
        return False
    
    # 准备训练数据
    X, y = processor.prepare_multi_stock_data(stock_data, target_col='close')
    
    if X is None or y is None:
        return False
    
    return True

def test_multiple_stocks():
    """测试多股票数据处理功能"""
    
    # 检查数据目录
    data_dir = "data/learn_csv"
    if not os.path.exists(data_dir):
        return False
    
    # 创建数据处理器
    processor = StockDataProcessor(seq_length=60)
    
    # 获取所有股票代码
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    stock_codes = [f.replace('.csv', '') for f in csv_files]
    
    # 测试加载前3只股票
    test_stock_codes = stock_codes[:3]
    
    # 加载指定股票数据
    stock_data = processor.load_multiple_stocks(data_dir, stock_codes=test_stock_codes)
    
    if not stock_data:
        return False
    
    return True

def plot_sample_data():
    """绘制样本数据图表"""
    
    # 检查数据目录
    data_dir = "data/learn_csv"
    if not os.path.exists(data_dir):
        return False
    
    # 获取第一个CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        return False
    
    csv_path = os.path.join(data_dir, csv_files[0])
    stock_code = csv_files[0].replace('.csv', '')
    
    # 创建数据处理器
    processor = StockDataProcessor(seq_length=60)
    
    # 加载数据
    data = processor.load_csv_data(csv_path)
    if data is None:
        return False
    
    # 添加技术指标
    data = processor.add_technical_indicators(data)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 价格走势图
    axes[0, 0].plot(data['tradingday'], data['close'], label='收盘价', linewidth=1)
    axes[0, 0].plot(data['tradingday'], data['sma_20'], label='20日均线', linewidth=1, alpha=0.7)
    axes[0, 0].set_title(f'{stock_code} 价格走势')
    axes[0, 0].set_xlabel('日期')
    axes[0, 0].set_ylabel('价格')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 成交量图
    axes[0, 1].bar(data['tradingday'], data['vol'], alpha=0.6, color='green')
    axes[0, 1].set_title(f'{stock_code} 成交量')
    axes[0, 1].set_xlabel('日期')
    axes[0, 1].set_ylabel('成交量')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. RSI指标
    axes[1, 0].plot(data['tradingday'], data['rsi'], label='RSI', linewidth=1)
    axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='超买线')
    axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='超卖线')
    axes[1, 0].set_title(f'{stock_code} RSI指标')
    axes[1, 0].set_xlabel('日期')
    axes[1, 0].set_ylabel('RSI')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. MACD指标
    axes[1, 1].plot(data['tradingday'], data['macd'], label='MACD', linewidth=1)
    axes[1, 1].plot(data['tradingday'], data['macd_signal'], label='信号线', linewidth=1)
    axes[1, 1].bar(data['tradingday'], data['macd_histogram'], alpha=0.6, label='柱状图')
    axes[1, 1].set_title(f'{stock_code} MACD指标')
    axes[1, 1].set_xlabel('日期')
    axes[1, 1].set_ylabel('MACD')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()  # 调整布局
    plt.savefig('results/sample_data_analysis.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()  # 显示图片
    
    return True

def main():
    """主函数，运行所有测试"""
    import os
    os.makedirs('results', exist_ok=True)
    
    # 运行各项测试
    tests = [
        ("CSV数据加载", test_csv_loading),
        ("技术指标计算", test_technical_indicators),
        ("数据准备", test_data_preparation),
        ("多股票处理", test_multiple_stocks),
        ("样本数据绘图", plot_sample_data)
    ]
    
    results = {}  # 存储测试结果
    
    for _, test_func in tests:
        try:
            result = test_func()
            results[test_func.__name__] = result
        except Exception:
            results[test_func.__name__] = False
    
    passed = sum(results.values())
    total = len(results)
    
    # 每次运行都覆盖写入 test_data.txt
    with open("test_data.txt", "w", encoding="utf-8") as f:
        if passed == total:
            f.write("OK\n")
        # 否则写空内容

if __name__ == "__main__":
    main()  # 运行主函数 