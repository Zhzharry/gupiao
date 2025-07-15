"""
简化的预测测试脚本
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from learn.models.transformer_model import StockTransformer
from data.data_processor import StockDataProcessor

def test_prediction():
    """测试预测功能"""
    print("🎯 开始测试预测功能...")
    
    # 配置参数
    config = {
        'seq_length': 20,
        'input_dim': 21,
        'd_model': 64,
        'nhead': 8,
        'num_layers': 2,
        'dropout': 0.1
    }
    
    # 创建数据处理器
    processor = StockDataProcessor(seq_length=config['seq_length'])
    
    # 查找测试数据
    test_dirs = ["../../data/test_csv", "../../data/learn_csv"]
    test_file = None
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
            if csv_files:
                test_file = os.path.join(test_dir, csv_files[0])
                print(f"📁 找到测试文件: {test_file}")
                break
    
    if test_file is None:
        print("❌ 未找到测试数据文件")
        return
    
    # 加载数据
    print("📊 正在加载数据...")
    data = processor.load_csv_data(test_file)
    if data is None:
        print("❌ 数据加载失败")
        return
    
    print(f"✅ 数据加载成功，样本数: {len(data)}")
    
    # 添加技术指标
    print("🔧 正在添加技术指标...")
    data_with_indicators = processor.add_technical_indicators(data)
    
    # 准备数据
    print("📋 正在准备数据...")
    X, y = processor.prepare_data(data_with_indicators, target_col='close')
    
    if X is None or y is None:
        print("❌ 数据准备失败")
        return
    
    print(f"✅ 数据准备成功: X.shape={X.shape}, y.shape={y.shape}")
    
    # 创建模型
    print("🏗️  正在创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        seq_len=config['seq_length'],
        output_dim=1,
        dropout=config['dropout']
    ).to(device)
    
    # 加载模型权重
    model_path = "../../models/best_model.pth"
    if os.path.exists(model_path):
        print("📦 正在加载模型权重...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 模型加载成功")
    else:
        print("⚠️  模型文件不存在，使用随机权重")
    
    # 进行预测
    print("🔮 正在进行预测...")
    with torch.no_grad():
        # 使用最后一个序列进行预测
        last_sequence = X[-1:].astype(np.float32)
        X_tensor = torch.FloatTensor(last_sequence).to(device)
        prediction = model(X_tensor).cpu().numpy()[0, 0]
    
    print(f"🎯 预测结果: {prediction:.2f}")
    print(f"📊 实际值: {y[-1]:.2f}")
    print(f"📈 误差: {abs(prediction - y[-1]):.2f}")
    
    print("✅ 测试完成！")

if __name__ == "__main__":
    test_prediction() 