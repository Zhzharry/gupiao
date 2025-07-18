import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
from collections import defaultdict
import glob

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
        self.metadata = []  # 存储元数据（股票代码、日期等）
        
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
                
                # 存储元数据
                self.metadata.append({
                    'secucode': secucode,
                    'date': stock_data.iloc[i+self.sequence_length]['tradingday']
                })
        
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
    def __init__(self, model_save_dir='models', results_dir='results'):
        self.model_save_dir = model_save_dir
        self.results_dir = results_dir
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 数据标准化器
        self.scaler = None
        self.model = None
    def load_test_data(self, test_data_dir='./data/test_csv'):
        """加载测试数据（1-5月份数据）"""
        print("加载测试数据...")
        
        # 添加目录存在性检查
        if not os.path.exists(test_data_dir):
            raise FileNotFoundError(f"测试数据目录不存在: {test_data_dir}")
        
        all_data = []
        csv_files = glob.glob(os.path.join(test_data_dir, '*.csv'))
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        for file in tqdm(csv_files, desc="加载测试CSV文件"):
            try:
                # 添加编码尝试
                try:
                    df = pd.read_csv(file, dtype={'tradingday': str})
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding='gbk', dtype={'tradingday': str})  # 尝试中文编码
                    
                if not df.empty:
                    print(f"加载文件 {file} 成功，记录数: {len(df)}")
                    all_data.append(df)
                else:
                    print(f"警告: 文件 {file} 为空")
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")
        
        if not all_data:
            raise ValueError("未找到有效的测试数据文件")
        
        # 合并时显示进度
        print("合并数据...")
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"合并后总记录数: {len(combined_data)}")
        
        # 预处理前记录数
        print("预处理前记录数:", len(combined_data))
        combined_data = self.preprocess_data(combined_data)
        print("预处理后记录数:", len(combined_data))
        
        return combined_data
    
    def load_actual_data(self, actual_data_dir='./data/Adjustment_csv'):
        """加载6月份实际数据"""
        print("加载6月份实际数据...")
        
        if not os.path.exists(actual_data_dir):
            raise FileNotFoundError(f"实际数据目录不存在: {actual_data_dir}")
        
        all_data = []
        csv_files = glob.glob(os.path.join(actual_data_dir, '*.csv'))
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        for file in tqdm(csv_files, desc="加载实际数据CSV文件"):
            try:
                # 尝试多种编码
                for encoding in ['utf-8', 'gbk', 'latin1']:
                    try:
                        df = pd.read_csv(file, encoding=encoding, dtype={'tradingday': str})
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df.empty:
                    print(f"警告: 文件 {file} 为空")
                    continue
                    
                # 检查日期列是否存在
                date_col = None
                for col in ['tradingday', 'date', 'datetime', '交易日期']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if not date_col:
                    print(f"警告: 文件 {file} 无日期列")
                    continue
                    
                # 转换日期格式
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    # 移除无效日期
                    df = df[df[date_col].notna()]
                except Exception as e:
                    print(f"文件 {file} 日期转换失败: {e}")
                    continue
                    
                all_data.append(df)
                
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")
        
        if not all_data:
            raise ValueError("未找到有效的实际数据文件")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"合并后总记录数: {len(combined_data)}")
        
        # 查找日期列
        date_col = None
        for col in ['tradingday', 'date', 'datetime', '交易日期']:
            if col in combined_data.columns:
                date_col = col
                break
        
        if not date_col:
            raise ValueError("数据中未找到日期列")
        
        # 打印日期范围
        print("数据日期范围:", combined_data[date_col].min(), "至", combined_data[date_col].max())
        
        # 筛选6月数据（所有年份）
        june_data = combined_data[combined_data[date_col].dt.month == 6]
        print(f"找到 {len(june_data)} 条6月份记录")
        
        if len(june_data) == 0:
            # 打印各月份数据量统计
            month_counts = combined_data[date_col].dt.month.value_counts().sort_index()
            print("各月份数据量统计:\n", month_counts)
            
            # 尝试放宽条件：包含"6"的日期（如2023-06或6月等）
            june_condition = (
                combined_data[date_col].astype(str).str.contains('-06-|/06/|6月|Jun|June', case=False)
            )
            june_data = combined_data[june_condition]
            print(f"放宽条件后找到 {len(june_data)} 条6月相关记录")
        
        return june_data
    
    def preprocess_data(self, data):
        """数据预处理"""
        # 记录初始数据量
        original_count = len(data)
        
        # 确保数据类型正确
        numeric_columns = ['preclose', 'open', 'high', 'low', 'close', 'vol', 'amount']
        for col in numeric_columns:
            if col in data.columns:
                # 先转换为字符串再转换为数字，避免混合类型问题
                data[col] = pd.to_numeric(data[col].astype(str), errors='coerce')
        
        # 仅移除全部特征为NA的行
        data = data.dropna(subset=numeric_columns, how='all')
        
        # 移除价格为0的记录但保留接近0的小数
        price_columns = ['preclose', 'open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0.001]  # 允许微小价格
        
        # 打印过滤信息
        filtered_count = original_count - len(data)
        print(f"数据预处理过滤了 {filtered_count} 条记录，保留 {len(data)} 条")
        
        return data
            
    def load_model(self, model_path='models/best_enhanced_model.pth'):
        """加载训练好的模型"""
        print("加载训练好的模型...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型检查点
        # 修改加载模型检查点的代码
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)        
        # 创建模型
        self.model = TransformerStockPredictor(
            input_dim=6,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            sequence_length=30,
            dropout=0.1
        ).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载标准化器
        self.scaler = checkpoint['scaler']
        
        self.model.eval()
        print("模型加载完成!")
        
        return self.model
    
    def create_prediction_sequences(self, data, sequence_length=30):
        """为预测创建序列数据"""
        print("创建预测序列...")
        
        feature_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
        
        # 标准化数据
        features = data[feature_columns].values
        if self.scaler is None:
            raise RuntimeError("标准化器scaler未加载，无法进行特征标准化。请检查模型文件是否包含scaler。")
        features_scaled = self.scaler.transform(features)
        
        # 更新数据
        data_scaled = data.copy()
        data_scaled[feature_columns] = features_scaled
        
        sequences = []
        metadata = []
        
        # 按股票代码分组
        for secucode in data_scaled['secucode'].unique():
            stock_data = data_scaled[data_scaled['secucode'] == secucode].sort_values('tradingday')
            
            if len(stock_data) < sequence_length:
                continue
            
            # 取最后30天的数据作为序列
            last_sequence = stock_data[feature_columns].values[-sequence_length:]
            sequences.append(last_sequence)
            
            # 存储元数据
            metadata.append({
                'secucode': secucode,
                'last_date': stock_data.iloc[-1]['tradingday'],
                'last_close': stock_data.iloc[-1]['close']
            })
        
        sequences = np.array(sequences, dtype=np.float32)
        
        print(f"创建了 {len(sequences)} 个预测序列")
        return sequences, metadata
    
    def predict_june_prices(self, test_data):
        """预测6月份的收盘价"""
        print("开始预测6月份收盘价...")
        
        # 创建预测序列
        sequences, metadata = self.create_prediction_sequences(test_data)
        
        # 创建数据加载器
        sequences_tensor = torch.tensor(sequences)
        dataset = torch.utils.data.TensorDataset(sequences_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        # 预测
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="预测中"):
                batch_sequences = batch[0].to(self.device)
                if self.model is None:
                    raise RuntimeError("模型未加载，无法进行预测。请检查模型文件。")
                batch_predictions = self.model(batch_sequences)
                predictions.extend(batch_predictions.cpu().numpy())
        
        # 反标准化预测结果
        predictions = np.array(predictions)
        dummy_data = np.zeros((len(predictions), 6))
        dummy_data[:, 3] = predictions  # close价格在第4列（索引3）
        if self.scaler is None:
            raise RuntimeError("标准化器scaler未加载，无法反标准化预测结果。请检查模型文件是否包含scaler。")
        predictions_unscaled = self.scaler.inverse_transform(dummy_data)[:, 3]
        
        # 创建预测结果DataFrame
        prediction_results = []
        for i, meta in enumerate(metadata):
            prediction_results.append({
                'secucode': meta['secucode'],
                'tradingday': meta['last_date'],
                'predicted_close': predictions_unscaled[i],
                'last_actual_close': meta['last_close'],
                'prediction_change': (predictions_unscaled[i] - meta['last_close']) / meta['last_close']
            })
        
        predictions_df = pd.DataFrame(prediction_results)
        # 只保留2025年6月的预测结果
        june_predictions = predictions_df[predictions_df['tradingday'].astype(str).str.startswith('202506')].copy()
        # 保存为6月份预测
        june_pred_path = os.path.join(self.results_dir, 'june_predictions.csv')
        june_predictions.to_csv(june_pred_path, index=False, encoding='utf-8')
        print(f"6月份预测结果已保存: {june_pred_path}")
        print(f"预测完成，共预测 {len(june_predictions)} 条6月数据")
        return june_predictions
    
    def evaluate_predictions(self, predictions_df, actual_data):
        """评估预测结果"""
        print("评估预测结果...")
        
        # 合并预测和实际数据
        # 计算6月份平均收盘价作为实际值
        actual_avg = actual_data.groupby('secucode')['close'].mean().reset_index()
        actual_avg.columns = ['secucode', 'actual_avg_close']
        
        # 合并数据
        evaluation_data = pd.merge(predictions_df, actual_avg, on='secucode', how='inner')
        
        if len(evaluation_data) == 0:
            print("警告：没有找到匹配的股票代码进行评估")
            return None
        
        # 计算评估指标
        predicted = evaluation_data['predicted_close'].values
        actual = evaluation_data['actual_avg_close'].values
        predicted = np.asarray(predicted)
        actual = np.asarray(actual)
        
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        
        # 计算相对误差
        relative_errors = np.abs((predicted - actual) / actual)
        mape = np.mean(relative_errors) * 100
        
        # 计算R²
        r2 = r2_score(actual, predicted)
        
        # 计算方向准确率
        predicted_direction = np.sign(evaluation_data['prediction_change'])
        actual_direction = np.sign((actual - evaluation_data['last_actual_close']) / evaluation_data['last_actual_close'])
        direction_accuracy = np.mean(predicted_direction == actual_direction)
        
        # 价格区间准确率
        def calculate_price_range_accuracy(predicted, actual, tolerance=0.05):
            """计算价格在容忍范围内的准确率"""
            within_range = np.abs((predicted - actual) / actual) <= tolerance
            return np.mean(within_range)
        
        price_accuracy_5 = calculate_price_range_accuracy(predicted, actual, 0.05)
        price_accuracy_10 = calculate_price_range_accuracy(predicted, actual, 0.10)
        price_accuracy_20 = calculate_price_range_accuracy(predicted, actual, 0.20)
        
        # 准备评估结果
        evaluation_results = {
            'total_stocks': len(evaluation_data),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'price_accuracy_5pct': price_accuracy_5,
            'price_accuracy_10pct': price_accuracy_10,
            'price_accuracy_20pct': price_accuracy_20,
            'mean_predicted_price': np.mean(predicted),
            'mean_actual_price': np.mean(actual),
            'std_predicted_price': np.std(predicted),
            'std_actual_price': np.std(actual)
        }
        
        print(f"评估完成，共评估 {len(evaluation_data)} 只股票")
        return evaluation_results, evaluation_data
    
    def generate_report(self, evaluation_results, evaluation_data):
        """生成详细的评估报告"""
        print("生成评估报告...")
        
        # 创建报告
        report = []
        report.append("="*80)
        report.append("股票价格预测模型评估报告")
        report.append("="*80)
        report.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"预测目标: 2025年6月份股票收盘价")
        report.append(f"评估股票数量: {evaluation_results['total_stocks']}")
        report.append("")
        
        # 基本统计指标
        report.append("1. 基本统计指标")
        report.append("-" * 40)
        report.append(f"均方误差 (MSE): {evaluation_results['mse']:.6f}")
        report.append(f"均方根误差 (RMSE): {evaluation_results['rmse']:.6f}")
        report.append(f"平均绝对误差 (MAE): {evaluation_results['mae']:.6f}")
        report.append(f"平均绝对百分比误差 (MAPE): {evaluation_results['mape']:.2f}%")
        report.append(f"决定系数 (R²): {evaluation_results['r2']:.6f}")
        report.append("")
        
        # 准确率指标
        report.append("2. 准确率指标")
        report.append("-" * 40)
        report.append(f"方向预测准确率: {evaluation_results['direction_accuracy']:.2%}")
        report.append(f"价格预测准确率 (±5%): {evaluation_results['price_accuracy_5pct']:.2%}")
        report.append(f"价格预测准确率 (±10%): {evaluation_results['price_accuracy_10pct']:.2%}")
        report.append(f"价格预测准确率 (±20%): {evaluation_results['price_accuracy_20pct']:.2%}")
        report.append("")
        
        # 价格统计
        report.append("3. 价格统计")
        report.append("-" * 40)
        report.append(f"预测价格均值: {evaluation_results['mean_predicted_price']:.2f}")
        report.append(f"实际价格均值: {evaluation_results['mean_actual_price']:.2f}")
        report.append(f"预测价格标准差: {evaluation_results['std_predicted_price']:.2f}")
        report.append(f"实际价格标准差: {evaluation_results['std_actual_price']:.2f}")
        report.append("")
        
        # 模型性能评估
        report.append("4. 模型性能评估")
        report.append("-" * 40)
        
        # 基于不同指标的评估
        if evaluation_results['r2'] > 0.8:
            r2_grade = "优秀"
        elif evaluation_results['r2'] > 0.6:
            r2_grade = "良好"
        elif evaluation_results['r2'] > 0.4:
            r2_grade = "一般"
        else:
            r2_grade = "较差"
        
        if evaluation_results['mape'] < 5:
            mape_grade = "优秀"
        elif evaluation_results['mape'] < 10:
            mape_grade = "良好"
        elif evaluation_results['mape'] < 20:
            mape_grade = "一般"
        else:
            mape_grade = "较差"
        
        if evaluation_results['direction_accuracy'] > 0.6:
            direction_grade = "优秀"
        elif evaluation_results['direction_accuracy'] > 0.55:
            direction_grade = "良好"
        elif evaluation_results['direction_accuracy'] > 0.5:
            direction_grade = "一般"
        else:
            direction_grade = "较差"
        
        report.append(f"R²评级: {r2_grade}")
        report.append(f"MAPE评级: {mape_grade}")
        report.append(f"方向准确率评级: {direction_grade}")
        report.append("")
        
        # 总体评估
        report.append("5. 总体评估")
        report.append("-" * 40)
        
        grades = [r2_grade, mape_grade, direction_grade]
        if grades.count('优秀') >= 2:
            overall_grade = "优秀"
        elif grades.count('良好') >= 2:
            overall_grade = "良好"
        elif grades.count('一般') >= 2:
            overall_grade = "一般"
        else:
            overall_grade = "较差"
        
        report.append(f"模型总体评级: {overall_grade}")
        report.append("")
        
        # 建议
        report.append("6. 改进建议")
        report.append("-" * 40)
        
        if evaluation_results['r2'] < 0.6:
            report.append("- 考虑增加更多技术指标作为特征")
            report.append("- 尝试调整模型架构或超参数")
        
        if evaluation_results['mape'] > 15:
            report.append("- 考虑对异常值进行更好的处理")
            report.append("- 可能需要更多的训练数据")
        
        if evaluation_results['direction_accuracy'] < 0.55:
            report.append("- 考虑使用分类模型来预测价格方向")
            report.append("- 增加市场情绪指标")
        
        report.append("")
        report.append("="*80)
        
        # 保存报告
        report_text = "\n".join(report)
        with open(os.path.join(self.results_dir, 'prediction_evaluation_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("评估报告已保存")
        return report_text
    
    def create_visualizations(self, evaluation_data):
        """创建可视化图表"""
        print("创建可视化图表...")
        
        # 设置字体，优先用系统默认字体，找不到SimHei/Arial Unicode MS时自动降级
        import matplotlib
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
        except Exception:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 预测vs实际散点图
        axes[0, 0].scatter(evaluation_data['actual_avg_close'], 
                          evaluation_data['predicted_close'], 
                          alpha=0.6, s=30)
        axes[0, 0].plot([evaluation_data['actual_avg_close'].min(), 
                        evaluation_data['actual_avg_close'].max()],
                       [evaluation_data['actual_avg_close'].min(), 
                        evaluation_data['actual_avg_close'].max()], 
                       'r--', lw=2)
        axes[0, 0].set_xlabel('实际收盘价')
        axes[0, 0].set_ylabel('预测收盘价')
        axes[0, 0].set_title('预测vs实际价格散点图')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差分布直方图
        errors = evaluation_data['predicted_close'] - evaluation_data['actual_avg_close']
        axes[0, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('预测误差')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('预测误差分布')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 相对误差分布
        relative_errors = np.abs((evaluation_data['predicted_close'] - evaluation_data['actual_avg_close']) / 
                                evaluation_data['actual_avg_close']) * 100
        axes[0, 2].hist(relative_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('相对误差 (%)')
        axes[0, 2].set_ylabel('频数')
        axes[0, 2].set_title('相对误差分布')
        axes[0, 2].axvline(x=5, color='red', linestyle='--', alpha=0.8, label='5%')
        axes[0, 2].axvline(x=10, color='orange', linestyle='--', alpha=0.8, label='10%')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 价格区间分布
        price_ranges = ['<10', '10-20', '20-50', '50-100', '>100']
        actual_counts = []
        predicted_counts = []
        
        for range_str in price_ranges:
            if range_str == '<10':
                actual_count = len(evaluation_data[evaluation_data['actual_avg_close'] < 10])
                predicted_count = len(evaluation_data[evaluation_data['predicted_close'] < 10])
            elif range_str == '10-20':
                actual_count = len(evaluation_data[(evaluation_data['actual_avg_close'] >= 10) & 
                                                 (evaluation_data['actual_avg_close'] < 20)])
                predicted_count = len(evaluation_data[(evaluation_data['predicted_close'] >= 10) & 
                                                    (evaluation_data['predicted_close'] < 20)])
            elif range_str == '20-50':
                actual_count = len(evaluation_data[(evaluation_data['actual_avg_close'] >= 20) & 
                                                 (evaluation_data['actual_avg_close'] < 50)])
                predicted_count = len(evaluation_data[(evaluation_data['predicted_close'] >= 20) & 
                                                    (evaluation_data['predicted_close'] < 50)])
            elif range_str == '50-100':
                actual_count = len(evaluation_data[(evaluation_data['actual_avg_close'] >= 50) & 
                                                 (evaluation_data['actual_avg_close'] < 100)])
                predicted_count = len(evaluation_data[(evaluation_data['predicted_close'] >= 50) & 
                                                    (evaluation_data['predicted_close'] < 100)])
            else:  # >100
                actual_count = len(evaluation_data[evaluation_data['actual_avg_close'] >= 100])
                predicted_count = len(evaluation_data[evaluation_data['predicted_close'] >= 100])
            
            actual_counts.append(actual_count)
            predicted_counts.append(predicted_count)
        
        x = np.arange(len(price_ranges))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, actual_counts, width, label='实际', alpha=0.8)
        axes[1, 0].bar(x + width/2, predicted_counts, width, label='预测', alpha=0.8)
        axes[1, 0].set_xlabel('价格区间')
        axes[1, 0].set_ylabel('股票数量')
        axes[1, 0].set_title('价格区间分布对比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(price_ranges)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 方向预测准确率
        predicted_direction = np.sign(evaluation_data['prediction_change'])
        actual_direction = np.sign((evaluation_data['actual_avg_close'] - evaluation_data['last_actual_close']) / 
                                  evaluation_data['last_actual_close'])
        
        direction_correct = (predicted_direction == actual_direction).sum()
        direction_wrong = len(evaluation_data) - direction_correct
        
        direction_labels = ['正确', '错误']
        direction_values = [direction_correct, direction_wrong]
        colors = ['green', 'red']
        
        axes[1, 1].pie(direction_values, labels=direction_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('方向预测准确率')
        
        # 6. 预测误差与实际价格的关系
        axes[1, 2].scatter(evaluation_data['actual_avg_close'], 
                          relative_errors, 
                          alpha=0.6, s=30)
        axes[1, 2].set_xlabel('实际收盘价')
        axes[1, 2].set_ylabel('相对误差 (%)')
        axes[1, 2].set_title('预测误差与实际价格的关系')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'prediction_evaluation_charts.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化图表已保存")
    
    def save_detailed_results(self, evaluation_data):
        """保存详细的预测结果"""
        print("保存详细结果...")
        
        # 计算额外的统计信息
        evaluation_data['absolute_error'] = np.abs(evaluation_data['predicted_close'] - evaluation_data['actual_avg_close'])
        evaluation_data['relative_error'] = np.abs((evaluation_data['predicted_close'] - evaluation_data['actual_avg_close']) / 
                                                  evaluation_data['actual_avg_close']) * 100
        evaluation_data['predicted_direction'] = np.sign(evaluation_data['prediction_change'])
        evaluation_data['actual_direction'] = np.sign((evaluation_data['actual_avg_close'] - evaluation_data['last_actual_close']) / 
                                                     evaluation_data['last_actual_close'])
        evaluation_data['direction_correct'] = (evaluation_data['predicted_direction'] == evaluation_data['actual_direction'])
        
        # 按相对误差排序
        evaluation_data_sorted = evaluation_data.sort_values('relative_error')
        
        # 保存完整结果
        evaluation_data_sorted.to_csv(os.path.join(self.results_dir, 'detailed_prediction_results.csv'), 
                                     index=False, encoding='utf-8')
        
        # 保存最好和最差的预测结果
        best_predictions = evaluation_data_sorted.head(20)
        worst_predictions = evaluation_data_sorted.tail(20)
        
        best_predictions.to_csv(os.path.join(self.results_dir, 'best_predictions.csv'), 
                               index=False, encoding='utf-8')
        worst_predictions.to_csv(os.path.join(self.results_dir, 'worst_predictions.csv'), 
                                index=False, encoding='utf-8')
        
        print("详细结果已保存")
    
    def run_prediction_evaluation(self):
        """运行完整的预测评估流程"""
        print("="*80)
        print("股票价格预测模型评估开始")
        print("="*80)
        
        try:
            # 1. 加载模型和数据
            self.load_model()
            test_data = self.load_test_data()
            actual_data = self.load_actual_data()
    
            # 2. 检查数据是否有效
            if test_data.empty or actual_data.empty:
                print("错误：测试数据或实际数据为空")
                return None, None
    
            # 3. 执行预测
            predictions_df = self.predict_june_prices(test_data)
            if predictions_df.empty:
                print("错误：未生成任何预测结果")
                return None, None
    
            # 4. 评估并处理空结果
            evaluation_results, evaluation_data = self.evaluate_predictions(predictions_df, actual_data)
            if not evaluation_results or evaluation_data.empty:
                print("评估失败：无有效数据")
                return None, None
    
            # 5. 生成报告和可视化
            self.generate_report(evaluation_results, evaluation_data)
            self.create_visualizations(evaluation_data)
            return evaluation_results, evaluation_data
            
            # 6. 生成报告
            report = self.generate_report(evaluation_results, evaluation_data)
            
            # 7. 创建可视化
            self.create_visualizations(evaluation_data)
            
            # 8. 保存详细结果
            self.save_detailed_results(evaluation_data)
            
            # 9. 保存评估结果JSON
            with open(os.path.join(self.results_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            print("="*80)
            print("预测评估完成!")
            print(f"评估报告保存在: {os.path.join(self.results_dir, 'prediction_evaluation_report.txt')}")
            print(f"可视化图表保存在: {os.path.join(self.results_dir, 'prediction_evaluation_charts.png')}")
            print(f"详细结果保存在: {os.path.join(self.results_dir, 'detailed_prediction_results.csv')}")
            print("="*80)
            
            # 打印简要结果
            print("\n简要评估结果:")
            print(f"评估股票数量: {evaluation_results['total_stocks']}")
            print(f"R²: {evaluation_results['r2']:.4f}")
            print(f"MAPE: {evaluation_results['mape']:.2f}%")
            print(f"方向准确率: {evaluation_results['direction_accuracy']:.2%}")
            print(f"价格准确率 (±5%): {evaluation_results['price_accuracy_5pct']:.2%}")
            print(f"价格准确率 (±10%): {evaluation_results['price_accuracy_10pct']:.2%}")
            
            return evaluation_results, evaluation_data
            
        except Exception as e:
            print(f"预测评估过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise

# 主程序
if __name__ == "__main__":
    # 创建预测评估器实例
    evaluator = StockPredictor()
    
    # 运行预测评估
    try:
        results, data = evaluator.run_prediction_evaluation()
        print("\n预测评估流程完成!")
        
        # 打印一些统计信息
        if results:
            print("\n=== 最终评估摘要 ===")
            print(f"总体评估: 基于R²={results['r2']:.4f}, MAPE={results['mape']:.2f}%, 方向准确率={results['direction_accuracy']:.2%}")
            
            if results['r2'] > 0.6 and results['mape'] < 15 and results['direction_accuracy'] > 0.55:
                print("🎉 模型表现良好！")
            elif results['r2'] > 0.4 and results['mape'] < 25 and results['direction_accuracy'] > 0.5:
                print("✅ 模型表现一般，但仍有参考价值")
            else:
                print("⚠️ 模型表现较差，建议进一步优化")
                
    except Exception as e:
        print(f"程序执行失败: {e}")
        print("请检查模型文件和数据文件是否存在，路径是否正确")