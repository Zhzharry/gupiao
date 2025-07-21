import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, sequence_length=60, model_save_path='models/'):
        self.sequence_length = sequence_length
        self.model_save_path = model_save_path
        self.scalers = {}
        self.models = {}
        
        # 创建模型保存目录
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    
    def load_and_preprocess_data(self, data_folder):
        """
        加载并预处理股票数据
        """
        print("正在加载股票数据...")
        
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"在 {data_folder} 目录下没有找到CSV文件")
        
        all_stock_data = {}
        
        for file_path in csv_files:
            try:
                # 提取股票代码
                stock_code = os.path.basename(file_path).replace('.csv', '')
                
                # 读取数据
                df = pd.read_csv(file_path)
                
                # 确保数据列名标准化
                df.columns = df.columns.str.lower()
                
                # 检查必需的列是否存在
                required_columns = ['tradingday', 'secucode', 'preclose', 'open', 'high', 'low', 'close', 'vol', 'amount', 'deals']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"警告：股票 {stock_code} 缺少列: {missing_columns}")
                    continue
                
                # 转换日期格式
                df['tradingday'] = pd.to_datetime(df['tradingday'].astype(str), format='%Y%m%d')
                df = df.sort_values('tradingday')
                
                # 重命名列以便后续处理
                df = df.rename(columns={
                    'tradingday': 'date',
                    'vol': 'volume'
                })
                
                # 添加基本价格特征
                df = self.add_price_features(df)
                
                # 添加技术指标
                df = self.add_technical_indicators(df)
                
                # 去除NaN值
                df = df.dropna()
                
                if len(df) < self.sequence_length:
                    print(f"警告：股票 {stock_code} 数据量不足，跳过")
                    continue
                
                all_stock_data[stock_code] = df
                print(f"已加载股票 {stock_code}，数据量: {len(df)}")
                
            except Exception as e:
                print(f"加载股票数据时出错 {file_path}: {e}")
                continue
        
        print(f"总共加载了 {len(all_stock_data)} 只股票的数据")
        return all_stock_data
    
    def add_price_features(self, df):
        """
        添加基本价格特征
        """
        # 价格相关特征
        df['price_change'] = df['close'] - df['preclose']  # 价格变动
        df['price_change_pct'] = (df['close'] - df['preclose']) / df['preclose']  # 价格变动百分比
        df['high_low_diff'] = df['high'] - df['low']  # 最高最低价差
        df['open_close_diff'] = df['close'] - df['open']  # 开盘收盘价差
        df['high_close_ratio'] = df['high'] / df['close']  # 最高价收盘价比率
        df['low_close_ratio'] = df['low'] / df['close']  # 最低价收盘价比率
        
        # 成交量相关特征
        df['vol_change'] = df['volume'].pct_change()  # 成交量变化率
        df['amount_change'] = df['amount'].pct_change()  # 成交额变化率
        df['avg_price'] = df['amount'] / df['volume']  # 平均成交价格
        df['deals_change'] = df['deals'].pct_change()  # 成交笔数变化率
        
        # 价量关系
        df['price_volume_trend'] = df['price_change_pct'] * df['vol_change']  # 价量趋势
        df['turnover_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()  # 换手率相对值
        
        return df
    
    def add_technical_indicators(self, df):
        """
        添加技术指标
        """
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # 移动平均线偏离度
        df['ma5_deviation'] = (df['close'] - df['ma5']) / df['ma5']
        df['ma20_deviation'] = (df['close'] - df['ma20']) / df['ma20']
        
        # RSI指标
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD指标
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # KDJ指标
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['kdj_k'] = rsv.ewm(alpha=1/3).mean()
        df['kdj_d'] = df['kdj_k'].ewm(alpha=1/3).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # 威廉指标
        df['williams_r'] = (high_max - df['close']) / (high_max - low_min) * (-100)
        
        # 成交量指标
        df['vol_ma5'] = df['volume'].rolling(window=5).mean()
        df['vol_ma20'] = df['volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma20']
        
        # OBV指标
        df['obv'] = (np.where(df['close'] > df['close'].shift(1), df['volume'], 
                     np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))).cumsum()
        
        return df
    
    def prepare_sequences(self, df, target_columns=['open', 'high', 'low', 'close']):
        """
        准备序列数据用于LSTM训练（多目标预测OHLC）
        """
        # 选择特征列
        feature_columns = [
            'preclose', 'open', 'high', 'low', 'close', 'volume', 'amount', 'deals',
            'price_change', 'price_change_pct', 'high_low_diff', 'open_close_diff',
            'high_close_ratio', 'low_close_ratio', 'vol_change', 'amount_change',
            'avg_price', 'deals_change', 'price_volume_trend', 'turnover_ratio',
            'ma5', 'ma10', 'ma20', 'ma5_deviation', 'ma20_deviation',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'kdj_k', 'kdj_d', 'kdj_j', 'williams_r',
            'vol_ma5', 'vol_ratio', 'obv'
        ]
        
        # 确保所有特征列都存在
        available_columns = [col for col in feature_columns if col in df.columns]
        print(f"使用特征列数量: {len(available_columns)}")
        
        # 准备特征数据
        feature_data = df[available_columns].values
        
        # 准备目标数据（OHLC）
        target_data = df[target_columns].values
        
        # 特征数据标准化
        feature_scaler = MinMaxScaler()
        scaled_features = feature_scaler.fit_transform(feature_data)
        
        # 目标数据标准化
        target_scaler = MinMaxScaler()
        scaled_targets = target_scaler.fit_transform(target_data)
        
        # 创建序列
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_targets[i])
        
        return np.array(X), np.array(y), feature_scaler, target_scaler
    
    def create_model(self, input_shape, output_dim=4):
        """
        创建LSTM模型（多输出：OHLC）
        """
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(output_dim, activation='linear')  # 输出OHLC四个值
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_individual_stock(self, stock_code, df):
        """
        训练单只股票的模型
        """
        print(f"正在训练股票 {stock_code} 的模型...")
        
        try:
            # 准备数据
            X, y, feature_scaler, target_scaler = self.prepare_sequences(df)
            
            if len(X) == 0:
                print(f"股票 {stock_code} 数据不足，跳过训练")
                return False
            
            # 保存标准化器
            self.scalers[stock_code] = {
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler
            }
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # 创建模型
            model = self.create_model((X.shape[1], X.shape[2]), output_dim=y.shape[1])
            
            # 设置回调函数
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=os.path.join(self.model_save_path, f'{stock_code}_model.h5'),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # 训练模型
            history = model.fit(
                X_train, y_train,
                epochs=150,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # 保存模型
            self.models[stock_code] = model
            
            # 评估模型
            train_loss = model.evaluate(X_train, y_train, verbose=0)
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            
            print(f"股票 {stock_code} 训练完成 - 训练损失: {train_loss[0]:.6f}, 测试损失: {test_loss[0]:.6f}")
            return True
            
        except Exception as e:
            print(f"训练股票 {stock_code} 时出错: {e}")
            return False
    
    def train_all_stocks(self, data_folder):
        """
        训练所有股票的模型
        """
        # 加载数据
        all_stock_data = self.load_and_preprocess_data(data_folder)
        
        successful_trainings = 0
        total_stocks = len(all_stock_data)
        
        for stock_code, df in all_stock_data.items():
            if self.train_individual_stock(stock_code, df):
                successful_trainings += 1
        
        print(f"\n训练完成！成功训练了 {successful_trainings}/{total_stocks} 只股票的模型")
        
        # 保存标准化器
        scaler_path = os.path.join(self.model_save_path, 'scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        print(f"标准化器已保存到: {scaler_path}")
        
        # 保存股票列表
        stock_list_path = os.path.join(self.model_save_path, 'stock_list.pkl')
        with open(stock_list_path, 'wb') as f:
            pickle.dump(list(self.scalers.keys()), f)
        
        print(f"股票列表已保存到: {stock_list_path}")
        
        return successful_trainings


def main():
    """
    主函数
    """
    print("=== 股票预测模型训练程序 ===")
    print("数据格式: tradingday, secucode, preclose, open, high, low, close, vol, amount, deals")
    
    # 配置参数
    DATA_FOLDER = "training_data/"  # 2023-2024年股票数据文件夹
    MODEL_SAVE_PATH = "models/"     # 模型保存路径
    SEQUENCE_LENGTH = 60           # 序列长度（天数）
    
    # 检查数据文件夹是否存在
    if not os.path.exists(DATA_FOLDER):
        print(f"错误：数据文件夹 {DATA_FOLDER} 不存在")
        print("请确保将2023-2024年的股票CSV文件放在该目录下")
        return
    
    # 创建预测器实例
    predictor = StockPredictor(
        sequence_length=SEQUENCE_LENGTH,
        model_save_path=MODEL_SAVE_PATH
    )
    
    try:
        # 开始训练
        successful_count = predictor.train_all_stocks(DATA_FOLDER)
        
        if successful_count > 0:
            print(f"\n🎉 训练成功完成！")
            print(f"📊 成功训练了 {successful_count} 只股票的模型")
            print(f"💾 模型文件保存在: {MODEL_SAVE_PATH}")
            print(f"📋 模型可以预测OHLC（开盘、最高、最低、收盘）价格")
            print(f"🔮 可以使用 predict.py 进行预测")
        else:
            print("❌ 没有成功训练任何模型，请检查数据格式和文件")
            
    except Exception as e:
        print(f"训练过程中出现错误: {e}")


if __name__ == "__main__":
    main()