import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import math
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

class StockDataset(Dataset):
    """股票数据集类"""
    def __init__(self, data, sequence_length=30, target_column='close'):
        self.data = data
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
        
        # 准备特征和目标数据
        self.prepare_data()
    
    def prepare_data(self):
        """准备序列数据"""
        self.sequences = []
        self.targets = []
        
        # 按股票代码分组
        for secucode in self.data['secucode'].unique():
            stock_data = self.data[self.data['secucode'] == secucode].sort_values('tradingday')
            
            if len(stock_data) < self.sequence_length + 1:
                continue
                
            # 获取特征数据
            features = stock_data[self.feature_columns].values
            
            # 创建序列
            for i in range(len(features) - self.sequence_length):
                seq = features[i:i+self.sequence_length]
                target = features[i+self.sequence_length][self.feature_columns.index(self.target_column)]
                
                self.sequences.append(seq)
                self.targets.append(target)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerStockPredictor(nn.Module):
    """Transformer股票预测模型"""
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, sequence_length=30, dropout=0.1):
        super(TransformerStockPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, sequence_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # 投影到模型维度
        x = self.input_projection(x)  # (batch_size, sequence_length, d_model)
        
        # 转置以适应Transformer输入格式
        x = x.transpose(0, 1)  # (sequence_length, batch_size, d_model)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Transformer编码
        transformer_output = self.transformer_encoder(x)
        
        # 使用最后一个时间步的输出
        last_output = transformer_output[-1]  # (batch_size, d_model)
        
        # 预测输出
        output = self.output_projection(last_output)
        
        return output.squeeze(-1)

class StockPredictor:
    """股票预测器主类"""
    def __init__(self, data_dir='data', model_save_dir='models', results_dir='results'):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.results_dir = results_dir
        
        # 创建目录
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 数据标准化器
        self.scaler = StandardScaler()
        
        # 训练历史
        self.train_history = {'loss': [], 'val_loss': [], 'epoch': []}
        
    def load_data(self):
        """加载训练数据"""
        print("加载训练数据...")
        train_data_dir = os.path.join(self.data_dir, 'learn_csv')
        if not os.path.exists(train_data_dir):
            print(f"❌ 训练数据目录不存在: {train_data_dir}")
            print("已自动创建该目录，请将训练用csv文件放入此目录后重新运行。")
            os.makedirs(train_data_dir, exist_ok=True)
            raise FileNotFoundError(f"训练数据目录不存在: {train_data_dir}")
        all_data = []
        
        csv_files = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]
        csv_files.sort()
        
        for file in tqdm(csv_files, desc="加载CSV文件"):
            file_path = os.path.join(train_data_dir, file)
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")
        
        if not all_data:
            raise ValueError("未找到有效的训练数据文件")
        
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 数据预处理
        combined_data = self.preprocess_data(combined_data)
        
        print(f"加载完成，总计 {len(combined_data)} 条记录")
        return combined_data
    
    def preprocess_data(self, data):
        """数据预处理"""
        print("数据预处理...")
        
        # 确保数据类型正确
        numeric_columns = ['preclose', 'open', 'high', 'low', 'close', 'vol', 'amount']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 移除缺失值
        data = data.dropna()
        
        # 移除异常值（价格为0或负数的记录）
        price_columns = ['preclose', 'open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # 计算技术指标
        data = self.calculate_technical_indicators(data)
        
        return data
    
    def calculate_technical_indicators(self, data):
        """计算技术指标"""
        print("计算技术指标...")
        
        # 按股票代码分组计算指标
        for secucode in data['secucode'].unique():
            mask = data['secucode'] == secucode
            stock_data = data[mask].sort_values('tradingday')
            
            if len(stock_data) < 20:  # 至少需要20天数据
                continue
            
            # 计算收益率
            stock_data['returns'] = stock_data['close'].pct_change()
            
            # 计算移动平均线
            stock_data['ma_5'] = stock_data['close'].rolling(window=5).mean()
            stock_data['ma_10'] = stock_data['close'].rolling(window=10).mean()
            stock_data['ma_20'] = stock_data['close'].rolling(window=20).mean()
            
            # 计算RSI
            delta = stock_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            stock_data['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算MACD
            exp1 = stock_data['close'].ewm(span=12).mean()
            exp2 = stock_data['close'].ewm(span=26).mean()
            stock_data['macd'] = exp1 - exp2
            stock_data['macd_signal'] = stock_data['macd'].ewm(span=9).mean()
            
            # 更新原数据
            data.loc[mask, stock_data.columns] = stock_data
        
        return data
    
    def prepare_datasets(self, data, sequence_length=30, train_split=0.8):
        """准备训练和验证数据集"""
        print("准备数据集...")
        
        # 按时间排序
        data = data.sort_values('tradingday')
        
        # 分割训练和验证数据
        split_idx = int(len(data) * train_split)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        # 标准化数据
        feature_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
        train_features = train_data[feature_columns].values
        val_features = val_data[feature_columns].values
        
        # 拟合标准化器
        self.scaler.fit(train_features)
        
        # 标准化
        train_data_scaled = train_data.copy()
        val_data_scaled = val_data.copy()
        
        train_data_scaled[feature_columns] = self.scaler.transform(train_features)
        val_data_scaled[feature_columns] = self.scaler.transform(val_features)
        
        # 创建数据集
        train_dataset = StockDataset(train_data_scaled, sequence_length)
        val_dataset = StockDataset(val_data_scaled, sequence_length)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset, epochs=100, batch_size=128, 
                   learning_rate=0.001, patience=20):
        """训练模型"""
        print("开始训练模型...")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        model = TransformerStockPredictor(
            input_dim=6,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            sequence_length=30,
            dropout=0.1
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=10)
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_idx, (sequences, targets) in enumerate(tqdm(train_loader, 
                                                                 desc=f"Epoch {epoch+1}/{epochs}")):
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences, targets = sequences.to(self.device), targets.to(self.device)
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # 更新学习率
            scheduler.step(avg_val_loss)
            
            # 记录训练历史
            self.train_history['loss'].append(avg_train_loss)
            self.train_history['val_loss'].append(avg_val_loss)
            self.train_history['epoch'].append(epoch + 1)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.8f}")
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_val_loss,
                    'scaler': self.scaler
                }, os.path.join(self.model_save_dir, 'best_enhanced_model.pth'))
                
                print(f"保存最佳模型 (验证损失: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
            
            print("-" * 50)
        
        print("训练完成!")
        return model
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['epoch'], self.train_history['loss'], 
                label='训练损失', color='blue')
        plt.plot(self.train_history['epoch'], self.train_history['val_loss'], 
                label='验证损失', color='red')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_history['epoch'], self.train_history['loss'], 
                label='训练损失', color='blue')
        plt.plot(self.train_history['epoch'], self.train_history['val_loss'], 
                label='验证损失', color='red')
        plt.title('训练和验证损失 (对数坐标)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, model, test_dataset):
        """评估模型"""
        print("评估模型...")
        
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        model.eval()
        predictions = []
        actual_values = []
        
        with torch.no_grad():
            for sequences, targets in tqdm(test_loader, desc="评估中"):
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                outputs = model(sequences)
                
                predictions.extend(outputs.cpu().numpy())
                actual_values.extend(targets.cpu().numpy())
        
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        # 计算评估指标
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)
        
        # 计算方向准确率
        actual_direction = np.sign(np.diff(actual_values))
        pred_direction = np.sign(np.diff(predictions))
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        print(f"评估结果:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"方向准确率: {direction_accuracy:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'predictions': predictions,
            'actual_values': actual_values
        }
    
    def save_training_config(self, config):
        """保存训练配置"""
        with open(os.path.join(self.results_dir, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def run_training(self):
        """运行完整的训练流程"""
        print("="*60)
        print("股票预测模型训练开始")
        print("="*60)
        
        try:
            # 加载数据
            data = self.load_data()
            
            # 准备数据集
            train_dataset, val_dataset = self.prepare_datasets(data)
            
            # 训练配置
            config = {
                'sequence_length': 30,
                'batch_size': 128,
                'learning_rate': 0.001,
                'epochs': 100,
                'patience': 20,
                'model_architecture': 'Transformer',
                'input_features': ['open', 'high', 'low', 'close', 'vol', 'amount'],
                'training_samples': len(train_dataset),
                'validation_samples': len(val_dataset)
            }
            
            # 保存配置
            self.save_training_config(config)
            
            # 训练模型
            model = self.train_model(
                train_dataset, val_dataset,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                patience=config['patience']
            )
            
            # 绘制训练历史
            self.plot_training_history()
            
            # 评估模型
            eval_results = self.evaluate_model(model, val_dataset)
            
            # 保存评估结果
            with open(os.path.join(self.results_dir, 'evaluation_results.json'), 'w') as f:
                eval_results_save = {k: v for k, v in eval_results.items() 
                                   if k not in ['predictions', 'actual_values']}
                json.dump(eval_results_save, f, indent=2, ensure_ascii=False)
            
            print("="*60)
            print("训练完成!")
            print(f"最佳模型保存在: {os.path.join(self.model_save_dir, 'best_enhanced_model.pth')}")
            print(f"训练历史图保存在: {os.path.join(self.results_dir, 'training_history.png')}")
            print("="*60)
            
            return model, eval_results
            
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            raise

# 主程序
if __name__ == "__main__":
    # 创建预测器实例
    predictor = StockPredictor()
    
    # 运行训练
    model, results = predictor.run_training()
    
    print("训练流程完成!")