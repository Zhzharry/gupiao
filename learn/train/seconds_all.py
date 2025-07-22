import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class StockDataDebugger:
    """股票数据调试器，用于诊断数据问题"""
    
    def __init__(self, data_path):
        self.data_path = data_path
    
    def diagnose_data_issues(self):
        """诊断数据问题"""
        print("=" * 60)
        print("开始诊断股票数据问题...")
        print("=" * 60)
        
        # 1. 检查数据目录是否存在
        print(f"1. 检查数据目录: {self.data_path}")
        if not os.path.exists(self.data_path):
            print(f"   ❌ 数据目录不存在!")
            return False
        else:
            print(f"   ✅ 数据目录存在")
        
        # 2. 检查CSV文件
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        print(f"\n2. 检查CSV文件:")
        print(f"   找到 {len(csv_files)} 个CSV文件")
        
        if len(csv_files) == 0:
            print("   ❌ 没有找到任何CSV文件!")
            # 检查是否有其他类型的文件
            all_files = os.listdir(self.data_path)
            print(f"   目录中的所有文件: {all_files[:10]}...")  # 只显示前10个
            return False
        else:
            print(f"   ✅ 找到CSV文件，前几个文件名:")
            for i, file in enumerate(csv_files[:5]):
                print(f"      - {os.path.basename(file)}")
            if len(csv_files) > 5:
                print(f"      ... 还有 {len(csv_files) - 5} 个文件")
        
        # 3. 检查文件内容
        print(f"\n3. 检查文件内容:")
        valid_files = 0
        invalid_files = []
        
        for i, csv_file in enumerate(csv_files[:10]):  # 只检查前10个文件避免输出太多
            try:
                stock_code = os.path.basename(csv_file).replace('.csv', '')
                print(f"   检查文件: {stock_code}")
                
                # 读取文件
                df = pd.read_csv(csv_file)
                print(f"      - 数据行数: {len(df)}")
                print(f"      - 列名: {list(df.columns)}")
                
                # 检查必需的列
                required_columns = ['tradingday', 'secucode', 'preclose', 'open', 
                                  'high', 'low', 'close', 'vol', 'amount', 'deals']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"      ❌ 缺少列: {missing_columns}")
                    invalid_files.append((stock_code, f"缺少列: {missing_columns}"))
                else:
                    print(f"      ✅ 所有必需列都存在")
                    
                    # 检查数据质量
                    print(f"      - 数据范围: {df['tradingday'].min()} 到 {df['tradingday'].max()}")
                    print(f"      - 收盘价范围: {df['close'].min():.2f} 到 {df['close'].max():.2f}")
                    
                    # 检查是否有足够的数据（增加到120天序列长度 + 10天预测期）
                    if len(df) >= 150:  
                        valid_files += 1
                        print(f"      ✅ 数据量充足")
                    else:
                        print(f"      ❌ 数据量不足 (需要至少150行)")
                        invalid_files.append((stock_code, f"数据量不足: 只有{len(df)}行"))
                
                print()  # 空行分隔
                
            except Exception as e:
                print(f"      ❌ 读取文件出错: {str(e)}")
                invalid_files.append((stock_code, f"读取错误: {str(e)}"))
        
        print(f"\n总结:")
        print(f"   - 总文件数: {len(csv_files)}")
        print(f"   - 检查的文件数: {min(10, len(csv_files))}")
        print(f"   - 有效文件数: {valid_files}")
        print(f"   - 无效文件数: {len(invalid_files)}")
        
        if invalid_files:
            print(f"\n无效文件详情:")
            for stock_code, reason in invalid_files:
                print(f"   - {stock_code}: {reason}")
        
        return valid_files > 0

class StockModelTrainer:
    def __init__(self, data_path, model_path):
        """
        初始化股票模型训练器
        
        Args:
            data_path: CSV数据文件路径
            model_path: 模型保存路径
        """
        self.data_path = data_path
        self.model_path = model_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
        # 修改参数：增加时间序列长度和预测窗口
        self.sequence_length = 120  # 使用过去120天的数据
        self.prediction_days = 10   # 预测未来10天的平均价格
        
        # 确保模型保存目录存在
        os.makedirs(self.model_path, exist_ok=True)
    
    def load_and_preprocess_data(self):
        """
        加载和预处理所有股票数据
        """
        print("开始加载股票数据...")
        
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        print(f"找到 {len(csv_files)} 个股票数据文件")
        
        if len(csv_files) == 0:
            print("❌ 没有找到任何CSV文件!")
            print(f"请检查路径: {self.data_path}")
            raise ValueError("没有找到CSV文件")
        
        all_data = []
        processed_stocks = 0
        error_details = []
        
        for csv_file in csv_files:
            try:
                # 读取CSV文件
                stock_code = os.path.basename(csv_file).replace('.csv', '')
                
                print(f"正在处理: {stock_code}...")
                df = pd.read_csv(csv_file)
                
                print(f"  - 原始数据行数: {len(df)}")
                print(f"  - 列名: {list(df.columns)}")
                
                # 检查必需的列
                required_columns = ['tradingday', 'secucode', 'preclose', 'open', 
                                  'high', 'low', 'close', 'vol', 'amount', 'deals']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    error_msg = f"缺少列: {missing_columns}"
                    print(f"  ❌ {error_msg}")
                    error_details.append((stock_code, error_msg))
                    continue
                
                # 数据预处理
                df = self.preprocess_single_stock(df, stock_code)
                print(f"  - 预处理后行数: {len(df)}")
                
                # 确保有足够的数据用于序列训练和预测目标计算
                min_required = self.sequence_length + self.prediction_days + 20
                if len(df) > min_required:  
                    all_data.append(df)
                    processed_stocks += 1
                    print(f"  ✅ 处理成功")
                    
                    if processed_stocks % 20 == 0:
                        print(f"已处理 {processed_stocks} 只股票...")
                else:
                    error_msg = f"数据量不足: 只有{len(df)}行，需要至少{min_required}行"
                    print(f"  ❌ {error_msg}")
                    error_details.append((stock_code, error_msg))
                
            except Exception as e:
                error_msg = f"处理错误: {str(e)}"
                print(f"  ❌ {error_msg}")
                error_details.append((stock_code, error_msg))
                continue
        
        print(f"\n处理完成:")
        print(f"- 成功处理: {processed_stocks} 只股票")
        print(f"- 失败数量: {len(error_details)} 只股票")
        
        if error_details:
            print(f"\n错误详情:")
            for stock_code, error in error_details[:10]:  # 只显示前10个错误
                print(f"  - {stock_code}: {error}")
            if len(error_details) > 10:
                print(f"  ... 还有 {len(error_details) - 10} 个错误")
        
        if not all_data:
            print("\n❌ 没有找到有效的股票数据!")
            print("可能的原因:")
            print("1. CSV文件格式不正确")
            print("2. 缺少必需的列")
            print("3. 数据量不足（每只股票需要足够的历史数据）")
            print("4. 数据中存在无法处理的错误")
            raise ValueError("没有找到有效的股票数据")
        
        # 合并所有股票数据
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"合并后的数据总量: {len(combined_data)} 条记录")
        
        return combined_data
    
    def preprocess_single_stock(self, df, stock_code):
        """
        预处理单只股票的数据，增强错误处理
        """
        try:
            # 转换交易日期 - 处理多种日期格式
            if df['tradingday'].dtype == 'object':
                # 尝试不同的日期格式
                try:
                    df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y%m%d')
                except:
                    try:
                        df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y-%m-%d')
                    except:
                        df['tradingday'] = pd.to_datetime(df['tradingday'])
            
            # 按日期排序
            df = df.sort_values('tradingday').reset_index(drop=True)
            
            # 检查和清理数值列
            numeric_columns = ['preclose', 'open', 'high', 'low', 'close', 'vol', 'amount', 'deals']
            for col in numeric_columns:
                if col in df.columns:
                    # 转换为数值类型，错误值设为NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除价格为0或负数的行
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    df = df[df[col] > 0]
            
            # 删除成交量为负的行
            if 'vol' in df.columns:
                df = df[df['vol'] >= 0]
            
            # 计算技术指标
            df = self.calculate_technical_indicators(df)
            
            # 处理缺失值 - 使用更稳健的方法
            df = df.fillna(method='forward').fillna(method='backward')
            
            # 删除仍然存在NaN的行
            df = df.dropna()
            
            # 添加股票代码列
            df['stock_code'] = stock_code
            
            return df
            
        except Exception as e:
            print(f"预处理股票 {stock_code} 时出错: {str(e)}")
            raise e
    
    def calculate_technical_indicators(self, df):
        """
        计算技术指标，增强错误处理
        """
        try:
            # 移动平均线（增加更多周期）
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['ma50'] = df['close'].rolling(window=50, min_periods=1).mean()
            
            # RSI指标
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)  # 避免除零错误
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 价格变化率（多个周期）
            df['price_change'] = df['close'].pct_change().fillna(0)
            df['price_change_3d'] = df['close'].pct_change(periods=3).fillna(0)
            df['price_change_5d'] = df['close'].pct_change(periods=5).fillna(0)
            df['volume_change'] = df['vol'].pct_change().fillna(0)
            
            # MACD指标
            exp1 = df['close'].ewm(span=12, min_periods=1).mean()
            exp2 = df['close'].ewm(span=26, min_periods=1).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # 布林带
            df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['close'].rolling(window=20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # 布林带位置（避免除零错误）
            bb_range = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = np.where(bb_range > 0, 
                                       (df['close'] - df['bb_lower']) / bb_range,
                                       0.5)
            
            # 波动率指标
            df['volatility'] = df['close'].rolling(window=20, min_periods=1).std()
            df['high_low_ratio'] = df['high'] / df['low']
            
            # 成交量相关指标
            df['volume_ma'] = df['vol'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['vol'] / (df['volume_ma'] + 1e-10)
            
            # 替换无穷大值
            df = df.replace([np.inf, -np.inf], np.nan)
            
            return df
            
        except Exception as e:
            print(f"计算技术指标时出错: {str(e)}")
            raise e
    
    def prepare_lstm_data(self, data):
        """
        准备LSTM模型的训练数据 - 修改为预测未来多天平均价
        """
        print("准备LSTM训练数据...")
        print(f"序列长度: {self.sequence_length}天")
        print(f"预测目标: 未来{self.prediction_days}天平均收盘价")
        
        # 选择更多特征列
        feature_columns = ['open', 'high', 'low', 'close', 'vol', 'amount', 'deals',
                          'ma5', 'ma10', 'ma20', 'ma50', 'rsi', 'price_change', 
                          'price_change_3d', 'price_change_5d', 'volume_change',
                          'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                          'volatility', 'high_low_ratio', 'volume_ratio']
        
        # 确保所有特征列都存在
        available_features = [col for col in feature_columns if col in data.columns]
        print(f"可用特征({len(available_features)}个): {available_features}")
        
        if not available_features:
            raise ValueError("没有可用的特征列")
        
        # 清理数据
        for col in available_features:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        
        # 删除缺失值
        data = data.dropna()
        
        min_required = self.sequence_length + self.prediction_days + 10
        if len(data) < min_required:
            raise ValueError(f"清理后的数据量不足: 只有{len(data)}行，需要至少{min_required}行")
        
        # 按股票分组处理，确保时间连续性
        X, y = [], []
        
        # 如果有股票代码列，按股票分组
        if 'stock_code' in data.columns:
            stock_groups = data.groupby('stock_code')
            print(f"处理 {len(stock_groups)} 只不同的股票")
            
            for stock_code, stock_data in stock_groups:
                stock_data = stock_data.sort_values('tradingday').reset_index(drop=True)
                
                if len(stock_data) < min_required:
                    continue
                
                # 提取特征和目标
                features = stock_data[available_features].values
                close_prices = stock_data['close'].values
                
                # 数据标准化（每只股票独立标准化）
                scaler_temp = MinMaxScaler(feature_range=(0, 1))
                features_scaled = scaler_temp.fit_transform(features)
                
                # 创建序列数据
                for i in range(self.sequence_length, len(features_scaled) - self.prediction_days + 1):
                    # 输入序列
                    X.append(features_scaled[i-self.sequence_length:i])
                    
                    # 目标：未来prediction_days天的平均收盘价
                    future_prices = close_prices[i:i+self.prediction_days]
                    y.append(np.mean(future_prices))
        
        else:
            # 如果没有股票代码列，作为单个时间序列处理
            features = data[available_features].values
            close_prices = data['close'].values
            
            # 数据标准化
            features_scaled = self.scaler.fit_transform(features)
            
            # 创建序列数据
            for i in range(self.sequence_length, len(features_scaled) - self.prediction_days + 1):
                X.append(features_scaled[i-self.sequence_length:i])
                future_prices = close_prices[i:i+self.prediction_days]
                y.append(np.mean(future_prices))
        
        X, y = np.array(X), np.array(y)
        
        if len(X) == 0:
            raise ValueError("无法创建有效的训练序列")
        
        print(f"特征数据形状: {X.shape}")
        print(f"目标数据形状: {y.shape}")
        
        # 目标值标准化
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 保存scaler
        scaler_path = os.path.join(self.model_path, 'feature_scaler.pkl')
        target_scaler_path = os.path.join(self.model_path, 'target_scaler.pkl')
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(target_scaler_path, 'wb') as f:
            pickle.dump(target_scaler, f)
        
        # 分割训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
        
        print(f"训练数据形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"测试数据形状: X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, target_scaler
    
    def build_lstm_model(self, input_shape):
        """构建更深层的LSTM模型"""
        model = Sequential([
            # 第一层LSTM
            LSTM(units=100, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # 第二层LSTM
            LSTM(units=100, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            # 第三层LSTM
            LSTM(units=50, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            # 第四层LSTM
            LSTM(units=50, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # 全连接层
            Dense(units=50, activation='relu'),
            Dropout(0.2),
            Dense(units=25, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])
        
        # 使用自适应学习率优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train_model(self):
        """训练股票预测模型"""
        print("开始训练股票预测模型...")
        print(f"配置: {self.sequence_length}天序列 → 未来{self.prediction_days}天平均价")
        
        # 加载和预处理数据
        data = self.load_and_preprocess_data()
        
        # 准备LSTM数据
        X_train, X_test, y_train, y_test, target_scaler = self.prepare_lstm_data(data)
        
        # 构建模型
        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        print("模型结构:")
        self.model.summary()
        
        # 设置回调函数
        checkpoint_path = os.path.join(self.model_path, 'best_stock_model.h5')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
        ]
        
        # 训练模型
        print("开始训练...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=200,  # 增加训练轮数
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型
        print("\n模型评估:")
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"训练集 - Loss: {train_loss[0]:.6f}, MAE: {train_loss[1]:.6f}, MAPE: {train_loss[2]:.2f}%")
        print(f"测试集 - Loss: {test_loss[0]:.6f}, MAE: {test_loss[1]:.6f}, MAPE: {test_loss[2]:.2f}%")
        
        # 保存完整模型
        model_path = os.path.join(self.model_path, 'stock_prediction_model.h5')
        self.model.save(model_path)
        
        # 保存训练配置
        config = {
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'model_architecture': 'LSTM',
            'input_features': X_train.shape[2]
        }
        
        config_path = os.path.join(self.model_path, 'model_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"\n模型训练完成！")
        print(f"模型文件: {model_path}")
        print(f"配置文件: {config_path}")
        print(f"预测目标: 基于{self.sequence_length}天历史数据预测未来{self.prediction_days}天平均价格")

def main():
    """主函数"""
    # 设置路径
    DATA_PATH = "/miniconda3/learn_transformer/learn/learn/train/data/learn_csv2"
    MODEL_PATH = "/miniconda3/learn_transformer/learn/learn/train/models2"
    
    print("=" * 80)
    print("股票预测模型训练程序（长序列 + 未来平均价预测版）")
    print("=" * 80)
    print(f"数据路径: {DATA_PATH}")
    print(f"模型保存路径: {MODEL_PATH}")
    print("改进内容:")
    print("• 序列长度: 60天 → 120天")
    print("• 预测目标: 单日价格 → 未来10天平均价格") 
    print("• 模型架构: 加深网络层数，添加批量归一化")
    print("• 技术指标: 增加更多特征维度")
    print("=" * 80)
    
    # 首先运行诊断
    debugger = StockDataDebugger(DATA_PATH)
    if not debugger.diagnose_data_issues():
        print("\n❌ 数据诊断失败，请解决上述问题后重试")
        return
    
    print("\n✅ 数据诊断通过，开始训练模型...")
    
    try:
        # 创建训练器实例
        trainer = StockModelTrainer(DATA_PATH, MODEL_PATH)
        
        # 开始训练
        trainer.train_model()
        
        print("=" * 80)
        print("🎉 训练完成！")
        print("=" * 80)
        print("模型特点:")
        print("• 使用120天历史数据作为输入序列")
        print("• 预测未来10天的平均收盘价")
        print("• 包含20+种技术指标特征")
        print("• 深层LSTM网络架构")
        print("• 支持多股票联合训练")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()