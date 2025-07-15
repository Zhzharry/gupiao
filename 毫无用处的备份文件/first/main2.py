"""
股票价格预测模型 - 主训练程序 (修复版)
====================================

修复内容：
1. 修复 train_test_split 返回值解包问题
2. 添加 scaler 保存和加载功能
3. 改进模型保存机制，同时保存 processor
4. 优化代码结构和错误处理

作者：AI Assistant
创建时间：2024年
"""

# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import warnings
import pickle  # 用于保存 processor
warnings.filterwarnings('ignore')
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from learn.models.transformer_model import StockTransformer, AdvancedStockTransformer, create_model
from data.data_processor import StockDataProcessor

class StockTrainer:
    """股票训练器类，负责模型训练和评估"""
    
    def __init__(self, config):
        """初始化训练器"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建必要的目录
        os.makedirs('../../models', exist_ok=True)
        os.makedirs('../../results', exist_ok=True)
        os.makedirs('../../logs', exist_ok=True)
        
    def load_data(self):
        """加载和处理训练数据"""
        print("📂 正在加载训练数据...")
        
        # 创建数据处理器实例
        print("🔧 正在创建数据处理器...")
        processor = StockDataProcessor(seq_length=self.config['seq_length'])
        print(f"✅ 数据处理器创建完成，序列长度: {self.config['seq_length']}")
        
        # 加载训练数据
        train_data_dir = "../../data/learn_csv"
        if os.path.exists(train_data_dir):
            print(f"📁 找到训练数据目录: {train_data_dir}")
            train_stock_data = processor.load_multiple_stocks(
                train_data_dir, 
                stock_codes=self.config.get('stock_codes', None)
            )
            print(f"✅ 成功加载了 {len(train_stock_data)} 只股票的训练数据")
        else:
            print(f"❌ 训练数据目录不存在: {train_data_dir}")
            return None, None, None
        
        # 准备训练数据
        print("🔄 正在准备训练数据...")
        X_train, y_train = processor.prepare_multi_stock_data(
            train_stock_data, 
            target_col=self.config['target_col']
        )
        
        if X_train is None:
            print("❌ 训练数据准备失败")
            return None, None, None
        
        print(f"✅ 训练数据准备完成")
        if X_train is not None and y_train is not None:
            print(f"   📊 特征数据形状: {X_train.shape if hasattr(X_train, 'shape') else 'unknown'}")
            print(f"   📊 标签数据形状: {y_train.shape if hasattr(y_train, 'shape') else 'unknown'}")
        else:
            print("   📊 数据形状: 数据为None")
            
        return X_train, y_train, processor
    
    def create_model(self):
        """创建Transformer模型"""
        model_type = self.config.get('model_type', 'basic')
        
        if model_type == 'basic':
            model = StockTransformer(
                input_dim=self.config['input_dim'],
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                num_layers=self.config['num_layers'],
                seq_len=self.config['seq_length'],
                output_dim=1,
                dropout=self.config['dropout']
            )
        elif model_type == 'advanced':
            model = AdvancedStockTransformer(
                input_dim=self.config['input_dim'],
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                num_layers=self.config['num_layers'],
                seq_len=self.config['seq_length'],
                output_dim=1,
                dropout=self.config['dropout']
            )
        else:
            model = create_model(model_type, **self.config)
        
        return model.to(self.device)
    
    def save_model_and_processor(self, model, processor, model_path="../../models/best_model.pth"):
        """保存模型和数据处理器"""
        try:
            # 保存模型状态字典
            torch.save(model.state_dict(), model_path)
            print(f"✅ 模型已保存到: {model_path}")
            
            # 保存数据处理器
            processor_path = model_path.replace('.pth', '_processor.pkl')
            with open(processor_path, 'wb') as f:
                pickle.dump(processor, f)
            print(f"✅ 数据处理器已保存到: {processor_path}")
            
            # 保存配置文件
            config_path = model_path.replace('.pth', '_config.pkl')
            with open(config_path, 'wb') as f:
                pickle.dump(self.config, f)
            print(f"✅ 配置文件已保存到: {config_path}")
            
        except Exception as e:
            print(f"❌ 保存模型时出错: {e}")
    
    def train(self, X_train, y_train, processor):
        """训练模型"""
        print("🚀 开始训练模型...")
        print("=" * 60)
        
        # 内存优化
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        # 修复：正确接收 train_test_split 的4个返回值
        print("📊 正在划分训练集和验证集...")
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )
        
        print(f"✅ 训练集大小: {len(X_train_split):,}")
        print(f"✅ 验证集大小: {len(X_val):,}")
        print(f"✅ 数据类型: X_train {type(X_train_split)}, y_train {type(y_train_split)}")
        
        # 创建数据加载器
        print("📦 正在创建数据加载器...")
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_split), 
            torch.FloatTensor(y_train_split)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"✅ 数据加载器创建完成，训练批次数量: {len(train_loader)}")
        
        # 创建模型
        print("🏗️  正在创建模型...")
        model = self.create_model()
        print(f"✅ 模型创建完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 定义损失函数和优化器
        print("⚙️  正在配置优化器...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=10,
        )
        print(f"✅ 优化器配置完成，学习率: {self.config['learning_rate']}")
        
        # 训练循环
        print("\n🎯 开始训练循环...")
        print("=" * 80)
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            model.train()
            train_loss = 0
            batch_count = 0
            
            print(f"\n📈 Epoch [{epoch+1:3d}/{self.config['epochs']}] 开始训练...")
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                if batch_count % 10 == 0:
                    print(f"   📊 批次 {batch_count:4d}/{len(train_loader):4d}, 当前损失: {loss.item():.6f}")
            
            # 验证阶段
            print(f"🔍 Epoch [{epoch+1:3d}] 开始验证...")
            model.eval()
            val_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X = val_X.to(self.device)
                    val_y = val_y.to(self.device)
                    val_outputs = model(val_X)
                    val_loss += criterion(val_outputs.squeeze(), val_y).item()
                    val_batch_count += 1
                val_loss = val_loss / val_batch_count
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型和处理器
                self.save_model_and_processor(model, processor)
                print(f"   💾 保存最佳模型 (验证损失: {val_loss:.6f})")
            else:
                patience_counter += 1
            
            # 打印进度
            print(f"   📊 Epoch [{epoch+1:3d}/{self.config['epochs']}] 完成")
            print(f"   🎯 训练损失: {avg_train_loss:.6f}")
            print(f"   🔍 验证损失: {val_loss:.6f}")
            print(f"   📈 学习率: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"   ⏰ 耐心计数: {patience_counter}/{self.config.get('patience', 20)}")
            print("-" * 60)
            
            # 早停
            if patience_counter >= self.config.get('patience', 20):
                print(f"🛑 早停在第 {epoch+1} 轮")
                break
        
        # 加载最佳模型
        print("\n🔄 正在加载最佳模型...")
        model.load_state_dict(torch.load("../../models/best_model.pth"))
        print(f"✅ 最佳模型加载完成 (验证损失: {best_val_loss:.6f})")
        
        # 绘制训练曲线
        print("📊 正在绘制训练曲线...")
        self.plot_training_curves(train_losses, val_losses)
        print("✅ 训练曲线已保存到 ../../results/training_curves.png")
        
        print("\n🎉 训练完成！")
        print("=" * 60)
        
        return model, processor
    
    def evaluate(self, model, X_test, y_test):
        """评估模型性能"""
        print("🔍 正在评估模型性能...")
        print(f"📊 测试样本数: {len(X_test):,}")
        
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = model(X_test_tensor).cpu().numpy().flatten()
        
        # 计算评估指标
        print("📈 正在计算评估指标...")
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # 计算方向准确率
        direction_accuracy = np.mean(
            np.sign(np.diff(y_test)) == np.sign(np.diff(predictions))
        )
        
        # 打印评估结果
        print("\n📊 模型评估结果:")
        print("=" * 40)
        print(f"🎯 MSE (均方误差): {mse:.6f}")
        print(f"📏 MAE (平均绝对误差): {mae:.6f}")
        print(f"📊 R² (决定系数): {r2:.6f}")
        print(f"📈 方向准确率: {direction_accuracy:.6f}")
        print("=" * 40)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'predictions': predictions
        }
    
    def plot_training_curves(self, train_losses, val_losses):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 绘制训练和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制对数尺度的损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Training Loss', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', alpha=0.7)
        plt.title('Training and Validation Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../../results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    # 设置CUDA环境变量和内存优化
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("🎯 股票价格预测模型训练程序")
    print("=" * 60)
    
    # 配置参数
    config = {
        'seq_length': 20,
        'input_dim': 21,
        'd_model': 64,
        'nhead': 8,
        'num_layers': 2,
        'dropout': 0.1,
        'batch_size': 8,
        'learning_rate': 0.001,
        'epochs': 30,
        'patience': 10,
        'weight_decay': 1e-5,
        'model_type': 'basic',
        'target_col': 'close',
        'stock_codes': None,
        'max_samples': 50000
    }
    
    print("📋 配置参数:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # 创建训练器
    print("🏗️  正在创建训练器...")
    trainer = StockTrainer(config)
    print("✅ 训练器创建完成")
    
    # 加载数据
    print("\n📊 正在加载训练数据...")
    X_train, y_train, processor = trainer.load_data()
    
    if X_train is None:
        print("❌ 数据加载失败，退出训练")
        return
    
    # 限制数据量以节省内存
    if X_train is not None and y_train is not None and len(X_train) > config['max_samples']:
        print(f"📊 数据量过大，限制为 {config['max_samples']:,} 个样本")
        indices = np.random.choice(len(X_train), config['max_samples'], replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    print(f"✅ 数据加载完成，训练样本数: {len(X_train):,}")
    
    # 训练模型
    print("\n🚀 开始模型训练...")
    model, processor = trainer.train(X_train, y_train, processor)
    
    # 评估模型 - 修复：正确接收4个返回值
    print("\n📈 正在评估模型性能...")
    _, X_test, _, y_test = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)
    results = trainer.evaluate(model, X_test, y_test)
    
    # 检查评估结果
    if results is None:
        print("❌ 模型评估失败")
        results = {'mse': 0.0, 'r2': 0.0}
    
    # 打印完成信息
    print("\n🎉 训练完成！")
    print("=" * 60)
    print(f"📁 模型保存在: ../../models/best_model.pth")
    print(f"📊 训练曲线保存在: ../../results/training_curves.png")
    print(f"📈 模型性能: MSE={results['mse']:.4f}, R²={results['r2']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()