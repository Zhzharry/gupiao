"""
股票价格预测模型 - 主训练程序
====================================

文件作用：
- 这是整个项目的入口文件，负责协调整个训练流程
- 包含StockTrainer类，负责模型训练、验证和评估
- 处理数据加载、模型创建、训练循环、性能评估等完整流程
- 支持GPU/CPU训练，包含早停、学习率调度等训练优化策略

主要功能：
1. 数据加载和预处理
2. 模型创建和配置
3. 训练循环管理
4. 验证和评估
5. 模型保存和结果可视化

使用方法：
python main.py

作者：AI Assistant
创建时间：2024年
"""

# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import DataLoader, TensorDataset  # 数据加载器
import numpy as np  # 数值计算库
from sklearn.model_selection import train_test_split  # 数据集分割
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 评估指标
import matplotlib.pyplot as plt  # 绘图库
import os  # 操作系统接口
import pandas as pd  # 数据处理库
from datetime import datetime  # 日期时间处理
import warnings  # 警告处理
warnings.filterwarnings('ignore')  # 忽略警告信息
import sys
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from learn.models.transformer_model import StockTransformer, AdvancedStockTransformer, create_model  # Transformer模型
from data.data_processor import StockDataProcessor  # 数据处理

class StockTrainer:
    """股票训练器类，负责模型训练和评估"""
    
    def __init__(self, config):
        """初始化训练器
        
        Args:
            config (dict): 配置参数字典
        """
        self.config = config  # 保存配置参数
        # 设置设备（GPU或CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建必要的目录
        os.makedirs('../../models', exist_ok=True)  # 模型保存目录
        os.makedirs('../../results', exist_ok=True)  # 结果保存目录
        os.makedirs('../../logs', exist_ok=True)  # 日志保存目录
        
    def load_data(self):
        """加载和处理训练数据
        
        Returns:
            tuple: (X_train, y_train, processor) 训练数据和数据处理器
        """
        print("📂 正在加载训练数据...")
        
        # 创建数据处理器实例
        print("🔧 正在创建数据处理器...")
        processor = StockDataProcessor(seq_length=self.config['seq_length'])
        print(f"✅ 数据处理器创建完成，序列长度: {self.config['seq_length']}")
        
        # 加载训练数据（2023-2024年）
        train_data_dir = "../../data/learn_csv"  # 训练数据目录
        if os.path.exists(train_data_dir):  # 检查目录是否存在
            print(f"📁 找到训练数据目录: {train_data_dir}")
            # 加载多个股票的数据
            print("📊 正在加载股票数据...")
            train_stock_data = processor.load_multiple_stocks(
                train_data_dir, 
                stock_codes=self.config.get('stock_codes', None)  # 指定股票代码，None表示加载所有
            )
            print(f"✅ 成功加载了 {len(train_stock_data)} 只股票的训练数据")
        else:
            print(f"❌ 训练数据目录不存在: {train_data_dir}")
            return None, None, None  # 返回三个None值
        
        # 准备训练数据
        print("🔄 正在准备训练数据...")
        X_train, y_train = processor.prepare_multi_stock_data(
            train_stock_data, 
            target_col=self.config['target_col']  # 预测目标列（通常是收盘价）
        )
        
        if X_train is None:  # 检查数据准备是否成功
            print("❌ 训练数据准备失败")
            return None, None, None
        
        print(f"✅ 训练数据准备完成")
        if X_train is not None and y_train is not None:
            print(f"   📊 特征数据形状: {X_train.shape if hasattr(X_train, 'shape') else 'unknown'}")
            print(f"   📊 标签数据形状: {y_train.shape if hasattr(y_train, 'shape') else 'unknown'}")
        else:
            print("   📊 数据形状: 数据为None")
            
        return X_train, y_train, processor  # 返回训练数据和处理器
    
    def create_model(self):
        """创建Transformer模型
        
        Returns:
            nn.Module: 创建的模型实例
        """
        model_type = self.config.get('model_type', 'basic')  # 获取模型类型
        
        if model_type == 'basic':  # 基本Transformer模型
            model = StockTransformer(
                input_dim=self.config['input_dim'],  # 输入特征维度
                d_model=self.config['d_model'],  # 模型维度
                nhead=self.config['nhead'],  # 注意力头数
                num_layers=self.config['num_layers'],  # Transformer层数
                seq_len=self.config['seq_length'],  # 序列长度
                output_dim=1,  # 输出维度（预测一个值）
                dropout=self.config['dropout']  # Dropout率
            )
        elif model_type == 'advanced':  # 高级Transformer模型
            model = AdvancedStockTransformer(
                input_dim=self.config['input_dim'],
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                num_layers=self.config['num_layers'],
                seq_len=self.config['seq_length'],
                output_dim=1,
                dropout=self.config['dropout']
            )
        else:  # 其他模型类型
            model = create_model(model_type, **self.config)
        
        return model.to(self.device)  # 将模型移动到指定设备
    
    def train(self, X_train, y_train, processor):
        """训练模型
        
        Args:
            X_train (np.ndarray): 训练特征数据
            y_train (np.ndarray): 训练标签数据
            processor (StockDataProcessor): 数据处理器
            
        Returns:
            tuple: (model, processor) 训练好的模型和数据处理器
        """
        print("🚀 开始训练模型...")
        print("=" * 60)
        
        # 内存优化
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        # 划分训练集和验证集（时间序列不随机打乱）
        print("📊 正在划分训练集和验证集...")
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )
        
        # 打印数据集大小
        print(f"✅ 训练集大小: {len(X_train_split):,}")
        print(f"✅ 验证集大小: {len(X_val):,}")
        # 显示数据类型
        print(f"✅ 数据类型: X_train {type(X_train_split)}, y_train {type(y_train_split)}")
        
        # 创建数据加载器（不预先移动到GPU以节省内存）
        print("📦 正在创建数据加载器...")
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_split), 
            torch.FloatTensor(y_train_split)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],  # 批次大小
            shuffle=True,  # 随机打乱
            pin_memory=True if torch.cuda.is_available() else False  # 内存优化
        )
        
        # 验证集数据（小批量处理以节省内存）
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
        criterion = nn.MSELoss()  # 均方误差损失
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'],  # 学习率
            weight_decay=self.config.get('weight_decay', 1e-5)  # 权重衰减
        )
        
        # 学习率调度器（当验证损失不下降时降低学习率）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',  # 监控最小值
            factor=0.5,  # 学习率衰减因子
            patience=10,  # 耐心值
        )
        print(f"✅ 优化器配置完成，学习率: {self.config['learning_rate']}")
        
        # 训练循环
        print("\n🎯 开始训练循环...")
        print("=" * 80)
        train_losses = []  # 训练损失记录
        val_losses = []  # 验证损失记录
        best_val_loss = float('inf')  # 最佳验证损失
        patience_counter = 0  # 早停计数器
        
        for epoch in range(self.config['epochs']):  # 训练轮数
            # 训练阶段
            model.train()  # 设置为训练模式
            train_loss = 0  # 初始化训练损失
            batch_count = 0  # 批次计数器
            
            # 显示进度条
            print(f"\n📈 Epoch [{epoch+1:3d}/{self.config['epochs']}] 开始训练...")
            
            for batch_X, batch_y in train_loader:  # 遍历每个批次
                # 移动数据到设备
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()  # 清零梯度
                outputs = model(batch_X)  # 前向传播
                loss = criterion(outputs.squeeze(), batch_y)  # 计算损失
                loss.backward()  # 反向传播
                
                # 梯度裁剪 (可选) - 防止梯度爆炸
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()  # 更新参数
                train_loss += loss.item()  # 累加损失
                batch_count += 1
                
                # 每10个批次显示一次进度
                if batch_count % 10 == 0:
                    print(f"   📊 批次 {batch_count:4d}/{len(train_loader):4d}, 当前损失: {loss.item():.6f}")
            
            # 验证阶段
            print(f"🔍 Epoch [{epoch+1:3d}] 开始验证...")
            model.eval()  # 设置为评估模式
            val_loss = 0
            val_batch_count = 0
            with torch.no_grad():  # 不计算梯度
                for val_X, val_y in val_loader:  # 小批量验证
                    val_X = val_X.to(self.device)
                    val_y = val_y.to(self.device)
                    val_outputs = model(val_X)  # 验证集预测
                    val_loss += criterion(val_outputs.squeeze(), val_y).item()  # 验证损失
                    val_batch_count += 1
                val_loss = val_loss / val_batch_count  # 平均验证损失
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)  # 记录训练损失
            val_losses.append(val_loss)  # 记录验证损失（已经是float）
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:  # 如果验证损失更小
                best_val_loss = val_loss  # 更新最佳验证损失
                patience_counter = 0  # 重置计数器
                # 保存最佳模型
                torch.save(model.state_dict(), f"../../models/best_model.pth")
                print(f"   💾 保存最佳模型 (验证损失: {val_loss:.6f})")
            else:
                patience_counter += 1  # 增加计数器
            
            # 打印进度
            print(f"   📊 Epoch [{epoch+1:3d}/{self.config['epochs']}] 完成")
            print(f"   🎯 训练损失: {avg_train_loss:.6f}")
            print(f"   🔍 验证损失: {val_loss:.6f}")
            print(f"   📈 学习率: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"   ⏰ 耐心计数: {patience_counter}/{self.config.get('patience', 20)}")
            print("-" * 60)
            
            # 早停
            if patience_counter >= self.config.get('patience', 20):  # 如果超过耐心值
                print(f"🛑 早停在第 {epoch+1} 轮")
                break
            
            # 早停检查
            if val_loss < best_val_loss:  # 如果验证损失更小
                best_val_loss = val_loss  # 更新最佳验证损失
                patience_counter = 0  # 重置计数器
                # 保存最佳模型
                torch.save(model.state_dict(), f"../../models/best_model.pth")
            else:
                patience_counter += 1  # 增加计数器
            
            # 打印进度
            if (epoch + 1) % 10 == 0:  # 每10轮打印一次
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 早停
            if patience_counter >= self.config.get('patience', 20):  # 如果超过耐心值
                print(f"早停在第 {epoch+1} 轮")
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
        """评估模型性能
        
        Args:
            model (nn.Module): 训练好的模型
            X_test (np.ndarray): 测试特征数据
            y_test (np.ndarray): 测试标签数据
            
        Returns:
            dict: 评估结果字典
        """
        print("🔍 正在评估模型性能...")
        print(f"📊 测试样本数: {len(X_test):,}")
        
        model.eval()  # 设置为评估模式
        with torch.no_grad():  # 不计算梯度
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)  # 转换为张量
            predictions = model(X_test_tensor).cpu().numpy().flatten()  # 获取预测结果
        
        # 计算评估指标
        print("📈 正在计算评估指标...")
        mse = mean_squared_error(y_test, predictions)  # 均方误差
        mae = mean_absolute_error(y_test, predictions)  # 平均绝对误差
        r2 = r2_score(y_test, predictions)  # 决定系数
        
        # 计算方向准确率（预测价格变动方向的准确率）
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
        """绘制训练曲线
        
        Args:
            train_losses (list): 训练损失列表
            val_losses (list): 验证损失列表
        """
        plt.figure(figsize=(12, 4))  # 创建图形
        
        # 绘制训练和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')  # 训练损失
        plt.plot(val_losses, label='Validation Loss')  # 验证损失
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
        plt.yscale('log')  # 对数尺度
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()  # 调整布局
        plt.savefig('../../results/training_curves.png', dpi=300, bbox_inches='tight')  # 保存图片
        plt.show()  # 显示图片
    
    def predict_next_price(self, model, processor, stock_code):
        """预测下一个价格
        
        Args:
            model (nn.Module): 训练好的模型
            processor (StockDataProcessor): 数据处理器
            stock_code (str): 股票代码
            
        Returns:
            float: 预测的下一个价格
        """
        # 加载测试数据（2025年）
        test_data_dir = "../../data/test_csv"
        if not os.path.exists(test_data_dir):
            print(f"测试数据目录不存在: {test_data_dir}")
            return None
        
        # 加载特定股票的最新数据
        csv_path = os.path.join(test_data_dir, f"{stock_code}.csv")
        if not os.path.exists(csv_path):
            print(f"股票 {stock_code} 的测试数据不存在")
            return None
        
        # 加载数据
        latest_data = processor.load_csv_data(csv_path)
        if latest_data is None:
            return None
        
        # 添加技术指标
        latest_data = processor.add_technical_indicators(latest_data)
        
        # 准备特征列
        feature_columns = [
            'open', 'high', 'low', 'close', 'vol', 'amount',
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram', 'rsi',
            'bb_upper', 'bb_lower', 'bb_middle', 'volume_ratio',
            'price_position', 'volatility'
        ]
        
        # 检查哪些特征列存在
        available_features = [col for col in feature_columns if col in latest_data.columns]
        
        if len(available_features) != self.config['input_dim']:
            print(f"特征维度不匹配: 期望 {self.config['input_dim']}, 实际 {len(available_features)}")
            return None
        
        # 获取数据值
        data_values = latest_data[available_features].dropna().values
        
        if len(data_values) < processor.seq_length:
            print("数据不足，无法预测")
            return None
        
        # 标准化数据
        data_scaled = processor.scaler.transform(data_values)
        
        # 获取最后一个序列
        last_sequence = data_scaled[-processor.seq_length:]
        
        # 转换为张量
        X_pred = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            prediction = model(X_pred).cpu().numpy()[0, 0]
        
        return prediction

def main():
    """主函数"""
    # 设置CUDA环境变量和内存优化
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA执行，便于调试
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 内存分配优化
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("🎯 股票价格预测模型训练程序")
    print("=" * 60)
    
    # 配置参数 - 内存优化版本
    config = {
        'seq_length': 20,           # 减少序列长度，降低内存使用
        'input_dim': 21,            # 保持特征维度
        'd_model': 64,              # 减少模型维度：64/8=8
        'nhead': 8,                 # 8个头，64/8=8
        'num_layers': 2,            # 进一步减少层数
        'dropout': 0.1,             # 保持
        'batch_size': 8,            # 大幅减少批次大小
        'learning_rate': 0.001,     # 保持
        'epochs': 30,               # 减少轮数
        'patience': 10,             # 早停耐心值
        'weight_decay': 1e-5,       # 保持
        'model_type': 'basic',      # 基础模型
        'target_col': 'close',      # 预测收盘价
        'stock_codes': None,        # 使用全部股票
        'max_samples': 50000        # 限制最大样本数
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
    
    if X_train is None:  # 检查数据加载是否成功
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
    
    # 评估模型（使用部分训练数据作为测试集）
    print("\n📈 正在评估模型性能...")
    X_test, y_test = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)
    results = trainer.evaluate(model, X_test, y_test)
    
    # 检查评估结果
    if results is None:
        print("❌ 模型评估失败")
        results = {'mse': 0.0, 'r2': 0.0}  # 默认值
    
    # 预测示例
    print("\n🔮 正在进行预测示例...")
    stock_codes = config.get('stock_codes', None)
    if stock_codes and len(stock_codes) > 0:  # 如果指定了股票代码
        stock_code = stock_codes[0]
        print(f"📊 预测股票 {stock_code} 的下一个价格...")
        next_price = trainer.predict_next_price(model, processor, stock_code)
        if next_price:
            print(f"🎯 预测的下一个收盘价: {next_price:.2f}")
    else:
        # 获取第一个可用的股票代码
        test_data_dir = "../../data/test_csv"
        if os.path.exists(test_data_dir):
            csv_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]
            if csv_files:
                stock_code = csv_files[0].replace('.csv', '')
                print(f"📊 预测股票 {stock_code} 的下一个价格...")
                next_price = trainer.predict_next_price(model, processor, stock_code)
                if next_price:
                    print(f"🎯 预测的下一个收盘价: {next_price:.2f}")
    
    # 打印完成信息
    print("\n🎉 训练完成！")
    print("=" * 60)
    print(f"📁 模型保存在: ../../models/best_model.pth")
    print(f"📊 训练曲线保存在: ../../results/training_curves.png")
    print(f"📈 模型性能: MSE={results['mse']:.4f}, R²={results['r2']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()  # 运行主函数 