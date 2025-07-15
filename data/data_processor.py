"""
股票价格预测模型 - 数据处理模块
================================

文件作用：
- 负责股票数据的加载、清洗、预处理和特征工程
- 处理按日期组织的CSV文件格式（每天一个文件，包含所有股票）
- 添加技术指标（移动平均、MACD、RSI、布林带等）
- 数据标准化和序列化处理

主要功能：
1. 数据加载：从CSV文件加载股票数据
2. 技术指标：计算各种技术分析指标
3. 数据清洗：处理缺失值、异常值
4. 特征工程：创建训练所需的特征矩阵
5. 序列化：将时间序列数据转换为模型输入格式

技术指标包括：
- 移动平均线（SMA 5/10/20）
- 指数移动平均线（EMA 12/26）
- MACD指标（MACD、信号线、柱状图）
- RSI相对强弱指标
- 布林带（上轨、下轨、中轨）
- 成交量比率
- 价格位置指标
- 波动率指标

使用方法：
- 创建StockDataProcessor实例
- 调用load_multiple_stocks()加载数据
- 调用prepare_multi_stock_data()准备训练数据

作者：AI Assistant
创建时间：2024年
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

class StockDataProcessor:
    def __init__(self, seq_length=60):
        self.seq_length = seq_length
        self.scaler = StandardScaler()
        self.scalers = {}  # 为每个特征列创建单独的标准化器
        
    def load_csv_data(self, csv_path):
        """加载CSV数据"""
        try:
            data = pd.read_csv(csv_path)
            # 确保列名正确
            expected_columns = ['tradingday', 'secucode', 'preclose', 'open', 'high', 'low', 'close', 'vol', 'amount', 'deals']
            
            # 检查列名
            if not all(col in data.columns for col in expected_columns):
                print(f"警告：CSV文件缺少预期列名，当前列名：{list(data.columns)}")
            
            # 转换日期格式
            data['tradingday'] = pd.to_datetime(data['tradingday'], format='%Y%m%d')
            
            # 按日期排序
            data = data.sort_values('tradingday')
            
            return data
        except Exception as e:
            print(f"加载CSV数据失败: {e}")
            return None
    
    def load_multiple_stocks(self, data_dir, stock_codes=None):
        """加载多个股票的数据 - 处理按日期组织的CSV文件"""
        print(f"📁 正在从目录加载数据: {data_dir}")
        
        # 获取所有CSV文件并按日期排序
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        csv_files.sort()  # 按文件名排序（日期格式）
        
        print(f"📊 找到 {len(csv_files)} 个CSV文件")
        
        # 用于存储每个股票的数据
        stock_data_dict = {}
        
        # 处理每个CSV文件
        for i, csv_file in enumerate(csv_files):
            if i % 10 == 0:  # 每处理10个文件打印一次进度
                print(f"📈 正在处理文件 {i+1}/{len(csv_files)}: {csv_file}")
                
            csv_path = os.path.join(data_dir, csv_file)
            daily_data = self.load_csv_data(csv_path)
            
            if daily_data is None or len(daily_data) == 0:
                print(f"⚠️  跳过空文件: {csv_file}")
                continue
            
            # 处理每个股票的数据
            for _, row in daily_data.iterrows():
                stock_code = row['secucode']
                
                # 如果指定了股票代码，只处理指定的
                if stock_codes and stock_code not in stock_codes:
                    continue
                
                # 如果这个股票还没有数据，初始化
                if stock_code not in stock_data_dict:
                    stock_data_dict[stock_code] = []
                
                # 添加这一天的数据
                stock_data_dict[stock_code].append(row)
        
        # 将每个股票的数据转换为DataFrame
        final_stock_data = {}
        stock_days_count = {}
        for stock_code, data_list in stock_data_dict.items():
            stock_days_count[stock_code] = len(data_list)
            if len(data_list) > 0:
                # 转换为DataFrame
                stock_df = pd.DataFrame(data_list)
                # 按日期排序
                stock_df = stock_df.sort_values('tradingday')
                # 重置索引
                stock_df = stock_df.reset_index(drop=True)
                # 只保留有足够数据的股票
                if len(stock_df) >= self.seq_length + 5:  # 降低要求：至少需要序列长度+5天的数据
                    final_stock_data[stock_code] = stock_df
                    print(f"✅ 股票 {stock_code}: {len(stock_df)} 天数据")
                else:
                    print(f"⚠️  股票 {stock_code}: 数据不足 ({len(stock_df)} 天)，跳过")
        print(f"🎯 成功加载了 {len(final_stock_data)} 只股票的数据")
        # 输出前10只股票的天数统计
        sorted_counts = sorted(stock_days_count.items(), key=lambda x: x[1], reverse=True)
        print("📊 股票天数统计（前10只）：")
        for code, days in sorted_counts[:10]:
            print(f"   股票 {code}: {days} 天")
        if len(final_stock_data) == 0:
            print("❌ 没有任何股票满足要求，全部股票天数统计如下：")
            for code, days in sorted_counts:
                print(f"   股票 {code}: {days} 天")
        return final_stock_data
    
    def add_technical_indicators(self, data):
        """添加技术指标"""
        # 价格变化率
        data['price_change'] = data['close'].pct_change()
        data['price_change_pct'] = data['price_change'] * 100
        
        # 移动平均线
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        
        # 指数移动平均线
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # 成交量指标
        data['volume_sma'] = data['vol'].rolling(window=20).mean()
        data['volume_ratio'] = data['vol'] / data['volume_sma']
        
        # 价格位置指标
        data['price_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # 波动率
        data['volatility'] = data['close'].rolling(window=20).std()
        
        return data
        
    def prepare_data(self, data, target_col='close'):
        """准备训练数据"""
        print(f"🔍 开始处理数据，原始数据长度: {len(data)}")
        
        # 选择特征列
        feature_columns = [
            'open', 'high', 'low', 'close', 'vol', 'amount',
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram', 'rsi',
            'bb_upper', 'bb_lower', 'bb_middle', 'volume_ratio',
            'price_position', 'volatility'
        ]
        
        # 检查哪些特征列存在
        available_features = [col for col in feature_columns if col in data.columns]
        
        if len(available_features) != len(feature_columns):
            missing_features = set(feature_columns) - set(available_features)
            print(f"⚠️  缺少特征列: {missing_features}")
            print(f"📊 可用特征: {len(available_features)}/{len(feature_columns)}")
            return None, None
        
        print(f"✅ 所有特征列都存在: {len(available_features)} 个特征")
        
        # 删除缺失值
        data_clean = data.dropna(subset=available_features + [target_col])
        print(f"🧹 清理后数据长度: {len(data_clean)} (删除了 {len(data) - len(data_clean)} 行缺失值)")
        
        if len(data_clean) < self.seq_length + 1:
            print(f"⚠️  数据不足，需要至少 {self.seq_length + 1} 天，实际只有 {len(data_clean)} 天")
            return None, None
        
        # 准备特征和标签
        X = data_clean[available_features].values
        y = data_clean[target_col].values
        
        print(f"📊 特征数据形状: {X.shape}, 标签数据形状: {y.shape}")
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建序列数据
        X_seq, y_seq = [], []
        for i in range(self.seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.seq_length:i])
            y_seq.append(y[i])
        
        X_final = np.array(X_seq)
        y_final = np.array(y_seq)
        
        print(f"🎯 最终序列数据: X={X_final.shape}, y={y_final.shape}")
        return X_final, y_final

    def prepare_multi_stock_data(self, stock_data_dict, target_col='close'):
        """准备多股票的训练数据"""
        print("🔄 正在准备多股票训练数据...")
        print(f"📊 总共需要处理 {len(stock_data_dict)} 只股票")
        
        all_X, all_y = [], []
        successful_stocks = 0
        failed_stocks = 0
        
        for i, (stock_code, data) in enumerate(stock_data_dict.items()):
            if i % 50 == 0:  # 每处理50只股票打印一次进度
                print(f"📈 处理进度: {i+1}/{len(stock_data_dict)}")
                
            print(f"📊 处理股票 {stock_code}...")
            
            # 添加技术指标
            data_with_indicators = self.add_technical_indicators(data)
            
            # 准备数据
            X, y = self.prepare_data(data_with_indicators, target_col)
            
            if X is not None and y is not None:
                all_X.append(X)
                all_y.append(y)
                successful_stocks += 1
                print(f"✅ 股票 {stock_code}: {len(X)} 个样本")
            else:
                failed_stocks += 1
                print(f"❌ 股票 {stock_code}: 数据准备失败")
        
        print(f"📊 处理完成统计:")
        print(f"   ✅ 成功: {successful_stocks} 只股票")
        print(f"   ❌ 失败: {failed_stocks} 只股票")
        
        if all_X:
            print(f"🎯 合并 {successful_stocks} 只股票的数据...")
            X_combined = np.vstack(all_X)
            y_combined = np.hstack(all_y)
            print(f"✅ 最终数据形状: X={X_combined.shape}, y={y_combined.shape}")
            print(f"📊 总训练样本数: {len(X_combined):,}")
            return X_combined, y_combined
        else:
            print("❌ 没有成功准备任何股票的数据")
            return None, None
    
    def get_latest_data(self, stock_data, days=100):
        """获取最新数据用于预测"""
        if stock_data is None or len(stock_data) == 0:
            return None
            
        # 获取最新的数据
        latest_data = stock_data.tail(days).copy()
        
        # 添加技术指标
        latest_data = self.add_technical_indicators(latest_data)
        
        return latest_data

# 使用示例
if __name__ == "__main__":
    processor = StockDataProcessor(seq_length=60)
    
    # 测试加载数据
    # data = processor.load_csv_data("data/learn_csv/000001.csv")
    # if data is not None:
    #     print("数据加载成功")
    #     print(data.head())
    #     print(f"数据形状: {data.shape}")
    
    # 测试多股票数据加载
    # stock_data = processor.load_multiple_stocks("data/learn_csv", ["000001", "000002"])
    # print(f"加载了 {len(stock_data)} 只股票的数据") 