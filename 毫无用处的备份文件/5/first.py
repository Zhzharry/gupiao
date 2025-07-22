import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class StockModelTrainer:
    def __init__(self):
        # 设置绝对路径
        self.data_path = "/home/user/miniconda3/learn_transformer/learn/learn/train/data/learn_csv2"
        self.model_path = "/home/user/miniconda3/learn_transformer/learn/learn/train/models2"
        
        # 创建模型保存目录
        os.makedirs(self.model_path, exist_ok=True)
        
        # 模型参数
        self.sequence_length = 60  # 使用60天的数据预测下一天
        self.features = ['open', 'high', 'low', 'close', 'vol', 'amount']
        self.target = 'close'
        
    def load_and_preprocess_data(self, file_path):
        """加载并预处理单只股票数据"""
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 确保列名正确
            expected_columns = ['tradingday', 'secucode', 'preclose', 'open', 
                              'high', 'low', 'close', 'vol', 'amount', 'deals']
            
            if not all(col in df.columns for col in expected_columns):
                print(f"警告: {file_path} 缺少必要的列")
                return None, None, None
            
            # 转换日期格式并排序
            df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y%m%d')
            df = df.sort_values('tradingday').reset_index(drop=True)
            
            # 选择特征列
            feature_data = df[self.features].values
            target_data = df[self.target].values
            
            # 检查数据质量
            if len(df) < self.sequence_length + 10:
                print(f"警告: {file_path} 数据量不足 ({len(df)} 行)")
                return None, None, None
            
            # 数据标准化
            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
            
            feature_scaled = feature_scaler.fit_transform(feature_data)
            target_scaled = target_scaler.fit_transform(target_data.reshape(-1, 1))
            
            return feature_scaled, target_scaled, (feature_scaler, target_scaler)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            return None, None, None
    
    def create_sequences(self, feature_data, target_data):
        """创建时间序列数据"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(feature_data)):
            X.append(feature_data[i-self.sequence_length:i])
            y.append(target_data[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """构建LSTM模型"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_single_stock(self, file_path):
        """训练单只股票的模型"""
        # 获取股票代码（文件名去掉扩展名）
        stock_code = os.path.splitext(os.path.basename(file_path))[0]
        print(f"开始训练股票 {stock_code}...")
        
        # 加载和预处理数据
        feature_data, target_data, scalers = self.load_and_preprocess_data(file_path)
        
        if feature_data is None:
            print(f"跳过股票 {stock_code}，数据处理失败")
            return False
        
        # 创建时间序列
        X, y = self.create_sequences(feature_data, target_data)
        
        if len(X) == 0:
            print(f"跳过股票 {stock_code}，序列数据为空")
            return False
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # 构建模型
        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # 设置回调函数
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                f"{self.model_path}/{stock_code}_best.h5",
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # 训练模型
        try:
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # 评估模型
            train_loss = model.evaluate(X_train, y_train, verbose=0)[0]
            test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
            
            print(f"股票 {stock_code} - 训练损失: {train_loss:.6f}, 测试损失: {test_loss:.6f}")
            
            # 保存模型和缩放器
            model.save(f"{self.model_path}/{stock_code}_model.h5")
            
            with open(f"{self.model_path}/{stock_code}_scalers.pkl", 'wb') as f:
                pickle.dump(scalers, f)
            
            # 保存训练历史
            with open(f"{self.model_path}/{stock_code}_history.pkl", 'wb') as f:
                pickle.dump(history.history, f)
            
            print(f"股票 {stock_code} 模型训练完成并保存")
            return True
            
        except Exception as e:
            print(f"训练股票 {stock_code} 时出错: {str(e)}")
            return False
    
    def train_all_stocks(self):
        """训练所有股票的模型"""
        # 获取所有CSV文件
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        if not csv_files:
            print(f"在 {self.data_path} 中未找到CSV文件")
            return
        
        print(f"找到 {len(csv_files)} 个股票数据文件")
        
        successful_trains = 0
        failed_trains = 0
        
        for i, file_path in enumerate(csv_files, 1):
            print(f"\n进度: {i}/{len(csv_files)}")
            
            if self.train_single_stock(file_path):
                successful_trains += 1
            else:
                failed_trains += 1
        
        print(f"\n训练完成!")
        print(f"成功训练: {successful_trains} 只股票")
        print(f"训练失败: {failed_trains} 只股票")
        print(f"模型保存路径: {self.model_path}")
    
    def predict_stock(self, stock_code, days_ahead=1):
        """使用训练好的模型进行预测"""
        try:
            # 加载模型和缩放器
            model = tf.keras.models.load_model(f"{self.model_path}/{stock_code}_model.h5")
            
            with open(f"{self.model_path}/{stock_code}_scalers.pkl", 'rb') as f:
                feature_scaler, target_scaler = pickle.load(f)
            
            # 加载最新数据
            file_path = os.path.join(self.data_path, f"{stock_code}.csv")
            df = pd.read_csv(file_path)
            
            # 预处理最新数据
            recent_data = df[self.features].tail(self.sequence_length).values
            recent_scaled = feature_scaler.transform(recent_data)
            
            # 准备预测数据
            X_pred = recent_scaled.reshape(1, self.sequence_length, len(self.features))
            
            # 进行预测
            pred_scaled = model.predict(X_pred)
            prediction = target_scaler.inverse_transform(pred_scaled)[0][0]
            
            return prediction
            
        except Exception as e:
            print(f"预测股票 {stock_code} 时出错: {str(e)}")
            return None

def main():
    """主函数"""
    trainer = StockModelTrainer()
    
    print("开始训练所有股票模型...")
    trainer.train_all_stocks()
    
    print("\n训练完成！")
    print("\n示例：预测股票价格")
    print("使用方法：")
    print("prediction = trainer.predict_stock('000001')  # 替换为实际的股票代码")

if __name__ == "__main__":
    main()