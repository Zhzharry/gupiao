import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# GPU显存按需分配，防止显存不足报错
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(e)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_all_data(data_folder, sequence_length):
    """
    加载所有CSV文件，合并为一个大DataFrame，按股票和时间排序
    """
    print("正在加载并合并所有股票数据...")
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not csv_files:
        raise ValueError(f"在 {data_folder} 目录下没有找到CSV文件")
    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()
            required_columns = ['tradingday', 'secucode', 'preclose', 'open', 'high', 'low', 'close', 'vol', 'amount', 'deals']
            if not all(col in df.columns for col in required_columns):
                print(f"警告：文件 {file_path} 缺少列，跳过")
                continue
            all_data.append(df)
        except Exception as e:
            print(f"读取文件时出错 {file_path}: {e}")
            continue
    if not all_data:
        raise ValueError("没有成功加载任何数据文件")
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['tradingday'] = pd.to_datetime(combined_df['tradingday'].astype(str), format='%Y%m%d')
    combined_df = combined_df.rename(columns={'vol': 'volume'})
    combined_df = combined_df.sort_values(['secucode', 'tradingday'])
    print(f"合并数据总量: {len(combined_df)} 条记录")
    return combined_df

def add_features(df):
    # 这里可根据需要添加特征，简单保留原始特征
    return df

def prepare_joint_sequences(df, sequence_length=60, target_columns=['open', 'high', 'low', 'close']):
    """
    所有股票合并后，按股票分组滑窗，合成统一训练集
    """
    feature_columns = [
        'preclose', 'open', 'high', 'low', 'close', 'volume', 'amount', 'deals'
    ]
    X, y = [], []
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    all_X, all_y = [], []
    for code, group in df.groupby('secucode'):
        group = group.sort_values('tradingday')
        group = add_features(group)
        features = group[feature_columns].values
        targets = group[target_columns].values
        if len(features) < sequence_length + 1:
            continue
        for i in range(sequence_length, len(features)):
            all_X.append(features[i-sequence_length:i])
            all_y.append(targets[i])
    all_X = np.array(all_X)
    all_y = np.array(all_y)
    # 统一归一化
    nsamples, slen, nfeat = all_X.shape
    all_X_2d = all_X.reshape(-1, nfeat)
    all_X_scaled = scaler_X.fit_transform(all_X_2d).reshape(nsamples, slen, nfeat)
    all_y_scaled = scaler_y.fit_transform(all_y)
    print(f"训练样本数: {all_X_scaled.shape[0]}")
    return all_X_scaled, all_y_scaled, scaler_X, scaler_y, feature_columns

def create_joint_model(input_shape, output_dim=4):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    print("=== 多股票联合训练模型程序 ===")
    DATA_FOLDER = "data/learn_csv/"  # 数据文件夹
    MODEL_SAVE_PATH = "models2/"     # 新模型保存路径
    SEQUENCE_LENGTH = 60
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ 错误：数据文件夹 {DATA_FOLDER} 不存在")
        return
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    # 1. 加载数据
    df = load_and_merge_all_data(DATA_FOLDER, SEQUENCE_LENGTH)
    # 2. 生成序列
    X, y, scaler_X, scaler_y, feature_columns = prepare_joint_sequences(df, SEQUENCE_LENGTH)
    # 3. 划分训练/验证集
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    # 4. 创建模型
    model = create_joint_model((X.shape[1], X.shape[2]), output_dim=y.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_PATH, 'joint_model.h5'), monitor='val_loss', save_best_only=True, verbose=1)
    ]
    # 5. 训练
    print("开始训练...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    # 6. 保存模型和scaler
    model.save(os.path.join(MODEL_SAVE_PATH, 'joint_model.h5'))
    with open(os.path.join(MODEL_SAVE_PATH, 'scaler_X.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(MODEL_SAVE_PATH, 'scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    with open(os.path.join(MODEL_SAVE_PATH, 'feature_columns.pkl'), 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"🎉 联合模型和scaler已保存到 {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()