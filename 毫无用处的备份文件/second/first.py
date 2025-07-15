"""
股票价格预测模型调优方案
========================

针对50%准确率问题的全面优化方案，包括：
1. 数据质量改进
2. 特征工程优化
3. 模型架构调整
4. 训练策略改进
5. 评估指标优化

主要改进点：
- 更好的数据预处理和特征工程
- 改进的损失函数（方向预测+价格预测）
- 更合理的模型架构
- 更好的训练策略
- 更全面的评估指标
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
try:
    import ta  # 技术指标库
except ImportError:
    print("警告: ta库未安装，将使用简单技术指标")
    ta = None
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

class ImprovedStockDataProcessor:
    """改进的数据处理器"""
    
    def __init__(self, seq_length=30, prediction_days=1):
        self.seq_length = seq_length
        self.prediction_days = prediction_days
        self.feature_scaler = RobustScaler()  # 使用RobustScaler，对异常值更稳健
        self.target_scaler = StandardScaler()
        self.feature_columns = []
        
    def add_advanced_features(self, df):
        """添加高级特征工程"""
        df = df.copy()
        
        # 基础价格特征
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        
        # 价格位置特征
        df['price_position_5d'] = (df['close'] - df['close'].rolling(5).min()) / (df['close'].rolling(5).max() - df['close'].rolling(5).min())
        df['price_position_20d'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        
        # 波动率特征
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # 成交量特征
        df['volume_ma_5'] = df['vol'].rolling(5).mean()
        df['volume_ma_20'] = df['vol'].rolling(20).mean()
        df['volume_ratio'] = df['vol'] / df['volume_ma_20']
        
        # 价格突破特征
        df['breakout_up'] = (df['close'] > df['close'].rolling(20).max().shift(1)).astype(int)
        df['breakout_down'] = (df['close'] < df['close'].rolling(20).min().shift(1)).astype(int)
        
        # 技术指标 - 使用简单计算方法
        # RSI计算
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD计算
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # 布林带计算
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        
        # 移动平均线
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # 趋势特征
        df['trend_5d'] = (df['close'] > df['sma_5']).astype(int)
        df['trend_20d'] = (df['close'] > df['sma_20']).astype(int)
        
        # 市场状态特征
        df['market_cap'] = df['close'] * df['vol']  # 假设市值
        df['price_volume_trend'] = df['close'] * df['vol']
        
        return df
    
    def create_labels(self, df):
        """创建多任务标签"""
        labels = {}
        
        # 价格预测标签（回归）
        labels['price'] = df['close'].shift(-self.prediction_days)
        
        # 方向预测标签（分类）
        future_return = df['close'].shift(-self.prediction_days) / df['close'] - 1
        labels['direction'] = (future_return > 0).astype(int)
        
        # 幅度预测标签（分类）
        labels['magnitude'] = pd.cut(future_return, 
                                   bins=[-np.inf, -0.02, 0.02, np.inf], 
                                   labels=[0, 1, 2])  # 0:下跌, 1:横盘, 2:上涨
        
        return labels
    
    def prepare_sequences(self, df, labels):
        """准备序列数据"""
        # 选择特征列 - 排除日期、代码等非数值列
        exclude_cols = ['date', 'code', 'tradingday', 'secucode', 'ts_code']  # 添加更多可能的非数值列名
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 进一步过滤：只保留数值类型的列
        numeric_cols = []
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_cols.append(col)
            else:
                print(f"跳过非数值列: {col} (类型: {df[col].dtype})")
        
        self.feature_columns = numeric_cols
        print(f"使用的特征列数量: {len(numeric_cols)}")
        
        # 只选择数值列
        df_numeric = df[numeric_cols].copy()
        
        # 移除包含NaN的行
        df_clean = df_numeric.dropna()
        print(f"清理后的数据形状: {df_clean.shape}")
        
        # 对齐标签
        for key in labels:
            labels[key] = labels[key].loc[df_clean.index]
        
        # 标准化特征 - 现在只处理数值数据
        features_scaled = self.feature_scaler.fit_transform(df_clean)
        
        # 创建序列
        X, y_price, y_direction, y_magnitude = [], [], [], []
        
        for i in range(len(features_scaled) - self.seq_length - self.prediction_days + 1):
            if not pd.isna(labels['price'].iloc[i + self.seq_length - 1]):
                X.append(features_scaled[i:i + self.seq_length])
                y_price.append(labels['price'].iloc[i + self.seq_length - 1])
                y_direction.append(labels['direction'].iloc[i + self.seq_length - 1])
                y_magnitude.append(labels['magnitude'].iloc[i + self.seq_length - 1])
        
        print(f"最终序列数量: {len(X)}")
        return np.array(X), np.array(y_price), np.array(y_direction), np.array(y_magnitude)

class ImprovedStockTransformer(nn.Module):
    """改进的Transformer模型"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, seq_len=30, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = self.create_positional_encoding(seq_len, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 多头输出
        self.price_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 上涨/下跌
        )
        
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # 下跌/横盘/上涨
        )
        
        # 注意力池化
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def create_positional_encoding(self, seq_len, d_model):
        """创建位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # [batch, seq, d_model]
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer编码
        x = x.transpose(0, 1)  # [seq+1, batch, d_model]
        x = self.transformer(x)
        
        # 使用CLS token的输出
        cls_output = x[0]  # [batch, d_model]
        
        # 多头输出
        price_output = self.price_head(cls_output)
        direction_output = self.direction_head(cls_output)
        magnitude_output = self.magnitude_head(cls_output)
        
        return price_output, direction_output, magnitude_output

class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha  # 价格预测权重
        self.beta = beta    # 方向预测权重
        self.gamma = gamma  # 幅度预测权重
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, price_pred, direction_pred, magnitude_pred, 
                price_true, direction_true, magnitude_true):
        
        # 价格预测损失
        price_loss = self.mse_loss(price_pred.squeeze(), price_true)
        
        # 方向预测损失
        direction_loss = self.ce_loss(direction_pred, direction_true.long())
        
        # 幅度预测损失（处理可能的NaN）
        valid_mask = ~torch.isnan(magnitude_true)
        if valid_mask.sum() > 0:
            magnitude_loss = self.ce_loss(magnitude_pred[valid_mask], magnitude_true[valid_mask].long())
        else:
            magnitude_loss = torch.tensor(0.0, device=price_pred.device)
        
        # 总损失
        total_loss = self.alpha * price_loss + self.beta * direction_loss + self.gamma * magnitude_loss
        
        return total_loss, price_loss, direction_loss, magnitude_loss

class ImprovedStockTrainer:
    """改进的股票训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = ImprovedStockDataProcessor(
            seq_length=config['seq_length'],
            prediction_days=config['prediction_days']
        )
        
    def train_model(self, train_data):
        """训练模型"""
        print("🚀 开始改进版模型训练...")
        
        # 数据预处理
        print("📊 数据预处理...")
        train_data = self.processor.add_advanced_features(train_data)
        labels = self.processor.create_labels(train_data)
        X, y_price, y_direction, y_magnitude = self.processor.prepare_sequences(train_data, labels)
        
        print(f"✅ 数据准备完成: {X.shape[0]} 个样本")
        print(f"   特征维度: {X.shape[2]}")
        print(f"   序列长度: {X.shape[1]}")
        
        # 数据划分
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_price_train, y_price_val = y_price[:split_idx], y_price[split_idx:]
        y_direction_train, y_direction_val = y_direction[:split_idx], y_direction[split_idx:]
        y_magnitude_train, y_magnitude_val = y_magnitude[:split_idx], y_magnitude[split_idx:]
        
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_price_train),
            torch.LongTensor(y_direction_train),
            torch.LongTensor(y_magnitude_train)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True
        )
        
        # 创建模型
        model = ImprovedStockTransformer(
            input_dim=X.shape[2],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            seq_len=self.config['seq_length'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # 损失函数和优化器
        criterion = MultiTaskLoss(alpha=1.0, beta=2.0, gamma=1.0)  # 强调方向预测
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.config['learning_rate'],
            steps_per_epoch=len(train_loader),
            epochs=self.config['epochs'],
            pct_start=0.3
        )
        
        # 训练循环
        train_losses = []
        val_accuracies = []
        best_val_acc = 0
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            model.train()
            epoch_losses = []
            
            for batch_X, batch_y_price, batch_y_direction, batch_y_magnitude in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y_price = batch_y_price.to(self.device)
                batch_y_direction = batch_y_direction.to(self.device)
                batch_y_magnitude = batch_y_magnitude.to(self.device)
                
                optimizer.zero_grad()
                
                price_pred, direction_pred, magnitude_pred = model(batch_X)
                
                total_loss, price_loss, direction_loss, magnitude_loss = criterion(
                    price_pred, direction_pred, magnitude_pred,
                    batch_y_price, batch_y_direction, batch_y_magnitude
                )
                
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_losses.append(total_loss.item())
            
            # 验证阶段
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                val_X = torch.FloatTensor(X_val).to(self.device)
                val_y_direction = torch.LongTensor(y_direction_val).to(self.device)
                
                # 批量预测以节省内存
                batch_size = self.config['batch_size']
                for i in range(0, len(val_X), batch_size):
                    batch_val_X = val_X[i:i+batch_size]
                    _, direction_pred, _ = model(batch_val_X)
                    val_predictions.extend(direction_pred.argmax(dim=1).cpu().numpy())
                    val_targets.extend(val_y_direction[i:i+batch_size].cpu().numpy())
            
            # 计算验证准确率
            val_acc = accuracy_score(val_targets, val_predictions)
            val_accuracies.append(val_acc)
            
                # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_improved_model.pth')
            
            # 打印进度
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            print(f'Epoch [{epoch+1}/{self.config["epochs"]}]')
            print(f'  Loss: {avg_loss:.4f}')
            print(f'  Val Accuracy: {val_acc:.4f}')
            print(f'  Best Val Accuracy: {best_val_acc:.4f}')
            print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)
        
        # 加载最佳模型
        model.load_state_dict(torch.load('best_improved_model.pth'))
        
        # 最终评估
        self.evaluate_model(model, X_val, y_price_val, y_direction_val, y_magnitude_val)
        
        return model
    
    def evaluate_model(self, model, X_test, y_price_test, y_direction_test, y_magnitude_test):
        """评估模型"""
        model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            
            # 批量预测
            price_predictions = []
            direction_predictions = []
            magnitude_predictions = []
            
            batch_size = self.config['batch_size']
            for i in range(0, len(X_test_tensor), batch_size):
                batch_X = X_test_tensor[i:i+batch_size]
                price_pred, direction_pred, magnitude_pred = model(batch_X)
                
                price_predictions.extend(price_pred.cpu().numpy().flatten())
                direction_predictions.extend(direction_pred.argmax(dim=1).cpu().numpy())
                magnitude_predictions.extend(magnitude_pred.argmax(dim=1).cpu().numpy())
        
        # 计算评估指标
        direction_acc = accuracy_score(y_direction_test, direction_predictions)
        direction_precision = precision_score(y_direction_test, direction_predictions, average='weighted')
        direction_recall = recall_score(y_direction_test, direction_predictions, average='weighted')
        direction_f1 = f1_score(y_direction_test, direction_predictions, average='weighted')
        
        # 价格预测评估
        price_mse = np.mean((y_price_test - price_predictions) ** 2)
        price_mae = np.mean(np.abs(y_price_test - price_predictions))
        
        print("\n📊 模型评估结果:")
        print("=" * 50)
        print(f"🎯 方向预测准确率: {direction_acc:.4f}")
        print(f"📈 方向预测精确率: {direction_precision:.4f}")
        print(f"📊 方向预测召回率: {direction_recall:.4f}")
        print(f"🎪 方向预测F1分数: {direction_f1:.4f}")
        print(f"💰 价格预测MSE: {price_mse:.6f}")
        print(f"💲 价格预测MAE: {price_mae:.6f}")
        print("=" * 50)
        
        return {
            'direction_accuracy': direction_acc,
            'direction_precision': direction_precision,
            'direction_recall': direction_recall,
            'direction_f1': direction_f1,
            'price_mse': price_mse,
            'price_mae': price_mae
        }

# 使用示例和配置
def main():
    """主函数 - 使用改进的配置"""
    
    print("🎯 股票价格预测模型训练程序")
    print("=" * 60)
    
    # 优化后的配置
    config = {
    # 输入序列参数
    'seq_length': 60,           # 扩大历史窗口（捕获更长依赖）
    'prediction_days': 7,        # 支持多步预测
    
    # 模型结构参数（核心调整）
    'd_model': 256,              # 原128→256（提升特征表达能力）
    'nhead': 8,                  # 保持与d_model兼容（256/8=32）
    'num_layers': 4,             # 原3→4（加深网络但避免过深）
    'dim_feedforward': 1024,     # FFN层维度（d_model的4倍）
    'dropout': 0.15,             # 原0.1→0.15（适度正则化）
    
    # 训练参数（适配扩大后的模型）
    'batch_size': 64,            # 原32→64（提升并行效率）
    'learning_rate': 5e-4,       # 原1e-3→5e-4（平衡速度和稳定性）
    'epochs': 150,               # 延长训练时间
    'weight_decay': 1e-5,        # 更小的L2约束（避免限制大模型）
    'gradient_clip': 0.5,        # 新增梯度裁剪（防梯度爆炸）
    
    # 可选扩展
    'use_learning_rate_scheduler': True,  # 启用学习率动态调整
    'patience': 10,              # 早停耐心值
}
    
    print("📋 配置参数:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # 创建训练器
    print("🏗️  正在创建训练器...")
    trainer = ImprovedStockTrainer(config)
    print("✅ 训练器创建完成")
    
    # 加载训练数据
    print("\n📊 正在加载训练数据...")
    train_data_dir = "../../data/learn_csv"
    
    if not os.path.exists(train_data_dir):
        print(f"❌ 训练数据目录不存在: {train_data_dir}")
        print("请确保数据已标准化并放在正确位置")
        return
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"❌ 在 {train_data_dir} 中没有找到CSV文件")
        return
    
    print(f"📁 找到 {len(csv_files)} 个CSV文件")
    
    # 加载第一个文件作为示例（你可以修改为加载多个文件）
    sample_file = csv_files[0]
    sample_path = os.path.join(train_data_dir, sample_file)
    print(f"📄 加载示例文件: {sample_file}")
    
    try:
        # 加载数据
        train_data = pd.read_csv(sample_path)
        print(f"✅ 成功加载数据，形状: {train_data.shape}")
        print(f"📊 数据列: {list(train_data.columns)}")
        
        # 检查必要的列
        required_cols = ['tradingday', 'secucode', 'close', 'open', 'high', 'low', 'vol', 'amount']
        missing_cols = [col for col in required_cols if col not in train_data.columns]
        if missing_cols:
            print(f"⚠️  缺少必要的列: {missing_cols}")
            print("请确保数据包含必要的价格和交易信息")
            return
        
        # 数据预处理
        print("\n🔧 数据预处理...")
        # 确保日期列格式正确
        if 'tradingday' in train_data.columns:
            train_data['tradingday'] = pd.to_datetime(train_data['tradingday'], format='%Y%m%d')
            train_data = train_data.sort_values('tradingday')
        
        # 移除重复数据
        train_data = train_data.drop_duplicates()
        print(f"✅ 预处理完成，最终数据形状: {train_data.shape if train_data is not None else 'None'}")
        
        # 开始训练
        print("\n🚀 开始模型训练...")
        model = trainer.train_model(train_data)
        
        print("\n🎉 训练完成！")
        print("=" * 60)
        print(f"📁 模型保存在: best_improved_model.pth")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {str(e)}")
        print("请检查数据格式和路径")
        return

if __name__ == "__main__":
    main()