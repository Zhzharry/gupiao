"""
股票价格预测程序 - 使用训练好的模型进行预测（修复版）
=======================================

功能：
1. 加载训练好的模型和数据处理器
2. 对新的股票数据进行预测
3. 可视化预测结果
4. 计算预测准确率和误差指标
5. 预测7月1号股市情况并生成对比报告

作者：AI Assistant
创建时间：2024年
修复时间：2025年
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import pickle
import warnings
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from learn.models.transformer_model import StockTransformer, AdvancedStockTransformer
from data.data_processor import StockDataProcessor

class StockPredictor:
    """股票价格预测器"""
    
    def __init__(self, model_path="../../models/best_model.pth"):
        """初始化预测器"""
        self.model_path = model_path
        self.processor_path = model_path.replace('.pth', '_processor.pkl')
        self.config_path = model_path.replace('.pth', '_config.pkl')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 创建结果目录
        os.makedirs('../../results/predictions', exist_ok=True)
        os.makedirs('../../data/Adjustment_csv', exist_ok=True)
        
        # 加载模型、处理器和配置
        self.model = None
        self.processor = None
        self.config = None
        
        self.load_model_and_processor()
        
    def load_model_and_processor(self):
        """加载模型、数据处理器和配置"""
        try:
            # 加载配置
            print("📋 正在加载配置文件...")
            if os.path.exists(self.config_path):
                with open(self.config_path, 'rb') as f:
                    self.config = pickle.load(f)
                print("✅ 配置文件加载成功")
            else:
                print("⚠️  配置文件不存在，使用默认配置")
                self.config = {
                    'seq_length': 20,
                    'input_dim': 21,
                    'd_model': 64,
                    'nhead': 8,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'model_type': 'basic'
                }
            
            # 加载数据处理器
            print("🔧 正在加载数据处理器...")
            if os.path.exists(self.processor_path):
                with open(self.processor_path, 'rb') as f:
                    self.processor = pickle.load(f)
                print("✅ 数据处理器加载成功")
            else:
                print("⚠️  数据处理器不存在，创建新的处理器")
                if self.config is not None:
                    self.processor = StockDataProcessor(seq_length=self.config['seq_length'])
                else:
                    self.processor = StockDataProcessor(seq_length=20)
            
            # 创建模型
            print("🏗️  正在创建模型...")
            if self.config is None:
                print("❌ 配置为空，无法创建模型")
                return
                
            model_type = self.config.get('model_type', 'basic')
            
            if model_type == 'basic':
                self.model = StockTransformer(
                    input_dim=self.config['input_dim'],
                    d_model=self.config['d_model'],
                    nhead=self.config['nhead'],
                    num_layers=self.config['num_layers'],
                    seq_len=self.config['seq_length'],
                    output_dim=1,
                    dropout=self.config['dropout']
                )
            elif model_type == 'advanced':
                self.model = AdvancedStockTransformer(
                    input_dim=self.config['input_dim'],
                    d_model=self.config['d_model'],
                    nhead=self.config['nhead'],
                    num_layers=self.config['num_layers'],
                    seq_len=self.config['seq_length'],
                    output_dim=1,
                    dropout=self.config['dropout']
                )
            
            if self.model is not None:
                self.model = self.model.to(self.device)
                print("✅ 模型创建成功")
                
                # 加载模型权重
                print("📦 正在加载模型权重...")
                if os.path.exists(self.model_path):
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    print("✅ 模型权重加载成功")
                else:
                    raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            else:
                raise ValueError("模型创建失败")
                
        except Exception as e:
            print(f"❌ 加载模型时出错: {e}")
            raise
    
    def prepare_single_stock_data(self, stock_data, target_col='close'):
        """
        准备单只股票的数据用于预测
        这是修复缺失方法的关键函数
        """
        try:
            # 检查配置是否存在
            if self.config is None:
                print("❌ 配置为空，无法准备数据")
                return None, None
                
            # 检查数据是否足够
            if len(stock_data) < self.config['seq_length']:
                print(f"❌ 数据不足，需要至少 {self.config['seq_length']} 条记录")
                return None, None
            
            # 准备特征列
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # 检查必要的列是否存在
            missing_cols = [col for col in feature_cols if col not in stock_data.columns]
            if missing_cols:
                print(f"❌ 缺少必要的列: {missing_cols}")
                return None, None
            
            # 创建技术指标
            data_with_indicators = self.create_technical_indicators(stock_data.copy())
            
            # 选择特征列 - 确保维度匹配模型配置
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'ma_5', 'ma_10', 'ma_20', 'ma_50',
                'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower',
                'price_change', 'volume_change',
                'volatility'  # 移除 'price_position' 以匹配21维
            ]
            
            # 检查特征列是否存在
            available_features = [col for col in feature_columns if col in data_with_indicators.columns]
            if len(available_features) < 5:
                print(f"⚠️  可用特征较少: {len(available_features)}")
                # 使用基础特征
                available_features = ['open', 'high', 'low', 'close', 'volume']
            
            # 提取特征数据
            feature_data = data_with_indicators[available_features].copy()
            
            # 数据标准化
            if self.processor is None or not hasattr(self.processor, 'scalers') or self.processor.scalers is None:
                print("⚠️  缺少预训练的标准化器，创建新的标准化器")
                if self.processor is None:
                    if self.config is not None:
                        self.processor = StockDataProcessor(seq_length=self.config['seq_length'])
                    else:
                        self.processor = StockDataProcessor(seq_length=20)
                self.processor.scalers = {}
                for col in feature_data.columns:
                    scaler = MinMaxScaler()
                    feature_data[col] = scaler.fit_transform(feature_data[col].values.reshape(-1, 1)).flatten()
                    self.processor.scalers[col] = scaler
            else:
                # 使用已有的标准化器
                for col in feature_data.columns:
                    if col in self.processor.scalers:
                        feature_data[col] = self.processor.scalers[col].transform(feature_data[col].values.reshape(-1, 1)).flatten()
                    else:
                        # 如果没有对应的标准化器，创建新的
                        scaler = MinMaxScaler()
                        feature_data[col] = scaler.fit_transform(feature_data[col].values.reshape(-1, 1)).flatten()
                        self.processor.scalers[col] = scaler
            
            # 创建序列数据
            X, y = [], []
            seq_length = self.config['seq_length']
            
            for i in range(seq_length, len(feature_data)):
                X.append(feature_data.iloc[i-seq_length:i].values)
                y.append(feature_data[target_col].iloc[i])
            
            X = np.array(X)
            y = np.array(y) if len(y) > 0 else None
            
            print(f"✅ 数据准备完成: X.shape={X.shape}, y.shape={y.shape if y is not None else 'None'}")
            
            return X, y
            
        except Exception as e:
            print(f"❌ 准备数据时出错: {e}")
            return None, None
    
    def create_technical_indicators(self, data):
        """创建技术指标"""
        try:
            # 移动平均线
            data['ma_5'] = data['close'].rolling(window=5).mean()
            data['ma_10'] = data['close'].rolling(window=10).mean()
            data['ma_20'] = data['close'].rolling(window=20).mean()
            data['ma_50'] = data['close'].rolling(window=50).mean()
            
            # 指数移动平均线
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            # 布林带
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            
            # 价格变化
            data['price_change'] = data['close'].pct_change()
            data['volume_change'] = data['volume'].pct_change()
            
            # 波动率
            data['volatility'] = data['close'].rolling(window=20).std()
            
            # 价格位置
            high_20 = data['high'].rolling(window=20).max()
            low_20 = data['low'].rolling(window=20).min()
            data['price_position'] = (data['close'] - low_20) / (high_20 - low_20)
            
            # 填充缺失值
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # 处理无穷大值
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # 如果还有无穷大值，用0填充
            data = data.replace([np.inf, -np.inf], 0)
            
            return data
            
        except Exception as e:
            print(f"❌ 创建技术指标时出错: {e}")
            return data
    
    def load_all_stock_data_from_daily_files(self, data_path="../../data/test_csv"):
        """从每日数据文件中加载所有股票的历史数据"""
        print(f"📂 正在从每日数据文件中加载股票历史数据: {data_path}")
        
        try:
            if not os.path.exists(data_path):
                print(f"❌ 数据路径不存在: {data_path}")
                return None
            
            # 获取所有CSV文件并按日期排序
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            csv_files.sort()  # 按文件名排序（日期）
            
            print(f"📅 找到 {len(csv_files)} 个交易日的数据文件")
            
            # 用于存储每只股票的历史数据
            stock_history = {}
            
            # 遍历每个交易日的数据文件
            for filename in csv_files:
                file_path = os.path.join(data_path, filename)
                trading_date = filename.replace('_daily.csv', '')
                
                try:
                    # 读取当日数据
                    daily_data = pd.read_csv(file_path)
                    
                    # 处理列名差异
                    column_mapping = {}
                    if 'vol' in daily_data.columns and 'volume' not in daily_data.columns:
                        column_mapping['vol'] = 'volume'
                    if 'tradingday' in daily_data.columns and 'date' not in daily_data.columns:
                        column_mapping['tradingday'] = 'date'
                    
                    # 重命名列
                    if column_mapping:
                        daily_data = daily_data.rename(columns=column_mapping)
                    
                    # 检查必要的列是否存在
                    required_cols = ['open', 'high', 'low', 'close', 'volume', 'secucode']
                    if not all(col in daily_data.columns for col in required_cols):
                        missing_cols = [col for col in required_cols if col not in daily_data.columns]
                        print(f"⚠️  跳过文件 {filename}: 缺少必要列 {missing_cols}")
                        continue
                    
                    # 为每只股票添加当日数据
                    for _, row in daily_data.iterrows():
                        stock_code = row['secucode']
                        
                        if stock_code not in stock_history:
                            stock_history[stock_code] = []
                        
                        # 添加当日数据
                        stock_history[stock_code].append({
                            'date': trading_date,
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume'],
                            'preclose': row.get('preclose', row['close']),
                            'amount': row.get('amount', 0),
                            'deals': row.get('deals', 0)
                        })
                        
                except Exception as e:
                    print(f"❌ 处理文件 {filename} 时出错: {e}")
                    continue
            
            # 将每只股票的数据转换为DataFrame
            stock_data = {}
            for stock_code, history in stock_history.items():
                if len(history) >= 20:  # 至少需要20天的数据
                    df = pd.DataFrame(history)
                    df = df.sort_values('date')  # 按日期排序
                    stock_data[stock_code] = df
                    print(f"✅ 股票 {stock_code}: {len(df)} 个交易日数据")
            
            print(f"✅ 成功加载 {len(stock_data)} 只股票的历史数据")
            return stock_data
            
        except Exception as e:
            print(f"❌ 加载股票历史数据时出错: {e}")
            return None
    
    def predict_single_stock_july_first(self, stock_data, stock_code):
        """预测单只股票7月1号的价格"""
        try:
            # 检查模型是否存在
            if self.model is None:
                print(f"❌ 模型为空，无法预测股票 {stock_code}")
                return None
            
            # 检查数据是否足够
            if stock_data is None or self.config is None or len(stock_data) < self.config['seq_length']:
                required_length = self.config['seq_length'] if self.config is not None else 20
                print(f"❌ 股票 {stock_code} 数据不足，需要至少 {required_length} 条记录")
                return None
            
            # 准备数据
            X, _ = self.prepare_single_stock_data(stock_data, target_col='close')
            
            if X is None or len(X) == 0:
                print(f"❌ 股票 {stock_code} 数据准备失败")
                return None
            
            # 使用最后一个序列进行预测
            last_sequence = X[-1]
            
            # 进行预测
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
                prediction = self.model(X_tensor).cpu().numpy().flatten()[0]
            
            # 反标准化预测结果
            if self.processor is not None and hasattr(self.processor, 'scalers') and 'close' in self.processor.scalers:
                prediction = self.processor.scalers['close'].inverse_transform(
                    np.array([prediction]).reshape(-1, 1)
                )[0, 0]
            
            # 获取最后一天的实际数据
            if len(stock_data) == 0:
                print(f"❌ 股票 {stock_code} 数据为空")
                return None
            last_day_data = stock_data.iloc[-1]
            
            # 构造预测结果
            result = {
                'stock_code': stock_code,
                'last_close': last_day_data['close'],
                'predicted_close': prediction,
                'predicted_open': prediction * 0.995,  # 估算开盘价
                'predicted_high': prediction * 1.01,   # 估算最高价
                'predicted_low': prediction * 0.99,    # 估算最低价
                'predicted_volume': last_day_data['volume'],  # 使用历史成交量
                'last_trading_day': last_day_data['date'],
                'data_points': len(stock_data)
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 预测股票 {stock_code} 时出错: {e}")
            return None
    
    def predict_all_stocks_july_first(self, data_path="../../data/test_csv"):
        """预测所有股票7月1号的价格"""
        print("🎯 开始预测所有股票7月1号的价格...")
        
        # 加载所有股票的历史数据
        stock_data = self.load_all_stock_data_from_daily_files(data_path)
        if stock_data is None:
            print("❌ 无法加载股票历史数据")
            return None
        
        # 预测每只股票
        predictions = {}
        total_stocks = len(stock_data)
        successful_predictions = 0
        
        print(f"\n📊 开始预测 {total_stocks} 只股票...")
        
        for i, (stock_code, data) in enumerate(stock_data.items(), 1):
            print(f"\n🔮 [{i}/{total_stocks}] 预测 {stock_code} 的7月1号价格...")
            
            result = self.predict_single_stock_july_first(data, stock_code)
            if result:
                predictions[stock_code] = result
                successful_predictions += 1
                print(f"✅ {stock_code}: 预测收盘价 {result['predicted_close']:.4f} (基于 {result['data_points']} 天数据)")
            else:
                print(f"❌ {stock_code}: 预测失败")
        
        print(f"\n✅ 完成预测，成功预测 {successful_predictions}/{total_stocks} 只股票")
        return predictions
    
    def save_july_first_predictions_to_csv(self, predictions, output_path="../../data/Adjustment_csv/7月1预测.csv"):
        """保存7月1号预测结果到CSV文件"""
        print("💾 正在保存7月1号预测结果到CSV文件...")
        
        try:
            # 创建预测数据列表
            prediction_data = []
            
            for stock_code, result in predictions.items():
                row = {
                    'tradingday': '20250701',
                    'secucode': stock_code,
                    'preclose': result['last_close'],
                    'open': result['predicted_open'],
                    'high': result['predicted_high'],
                    'low': result['predicted_low'],
                    'close': result['predicted_close'],
                    'vol': result['predicted_volume'],  # 使用 'vol' 而不是 'volume'
                    'amount': result['predicted_volume'] * result['predicted_close'],
                    'deals': 0  # 无法预测成交笔数
                }
                prediction_data.append(row)
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(prediction_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"✅ 预测结果已保存到: {output_path}")
            print(f"📊 共预测 {len(prediction_data)} 只股票")
            
            return output_path
            
        except Exception as e:
            print(f"❌ 保存预测结果时出错: {e}")
            return None
    
    def load_actual_july_first_data(self, actual_data_path="../../data/Adjustment_csv/20250701_daily.csv"):
        """加载7月1号的实际数据"""
        print(f"📂 正在加载实际数据: {actual_data_path}")
        
        try:
            if not os.path.exists(actual_data_path):
                print(f"❌ 实际数据文件不存在: {actual_data_path}")
                return None
            
            actual_data = pd.read_csv(actual_data_path)
            
            # 处理列名差异
            column_mapping = {}
            if 'vol' in actual_data.columns and 'volume' not in actual_data.columns:
                column_mapping['vol'] = 'volume'
            if 'tradingday' in actual_data.columns and 'date' not in actual_data.columns:
                column_mapping['tradingday'] = 'date'
            
            # 重命名列
            if column_mapping:
                actual_data = actual_data.rename(columns=column_mapping)
            
            print(f"✅ 成功加载实际数据，共 {len(actual_data)} 条记录")
            
            return actual_data
            
        except Exception as e:
            print(f"❌ 加载实际数据时出错: {e}")
            return None
    
    def compare_predictions_with_actual(self, predictions, actual_data):
        """比较预测结果与实际数据"""
        print("📊 正在比较预测结果与实际数据...")
        
        if actual_data is None:
            print("❌ 无实际数据可比较")
            return None
        
        # 创建比较结果列表
        comparison_results = []
        
        # 将实际数据转换为字典以便查找
        actual_dict = {}
        for _, row in actual_data.iterrows():
            actual_dict[row['secucode']] = row
        
        # 计算统计指标
        total_predictions = 0
        correct_direction = 0
        total_abs_error = 0
        total_rel_error = 0
        
        for stock_code, prediction in predictions.items():
            if stock_code in actual_dict:
                actual_row = actual_dict[stock_code]
                
                # 计算误差
                predicted_close = prediction['predicted_close']
                actual_close = actual_row['close']
                last_close = prediction['last_close']
                
                abs_error = abs(predicted_close - actual_close)
                rel_error = abs_error / actual_close * 100
                
                # 计算方向准确性
                predicted_direction = "上涨" if predicted_close > last_close else "下跌"
                actual_direction = "上涨" if actual_close > actual_row['preclose'] else "下跌"
                direction_correct = predicted_direction == actual_direction
                
                if direction_correct:
                    correct_direction += 1
                
                total_predictions += 1
                total_abs_error += abs_error
                total_rel_error += rel_error
                
                # 记录比较结果
                comparison_result = {
                    'stock_code': stock_code,
                    'predicted_close': predicted_close,
                    'actual_close': actual_close,
                    'last_close': last_close,
                    'abs_error': abs_error,
                    'rel_error': rel_error,
                    'predicted_direction': predicted_direction,
                    'actual_direction': actual_direction,
                    'direction_correct': direction_correct
                }
                comparison_results.append(comparison_result)
            else:
                print(f"⚠️  股票 {stock_code} 在实际数据中不存在")
        
        # 计算总体统计
        if total_predictions > 0:
            direction_accuracy = correct_direction / total_predictions
            avg_abs_error = total_abs_error / total_predictions
            avg_rel_error = total_rel_error / total_predictions
            
            summary = {
                'total_stocks': total_predictions,
                'direction_accuracy': direction_accuracy,
                'avg_abs_error': avg_abs_error,
                'avg_rel_error': avg_rel_error,
                'correct_direction_count': correct_direction
            }
            
            print(f"✅ 比较完成:")
            print(f"   总股票数: {total_predictions}")
            print(f"   方向准确率: {direction_accuracy:.2%}")
            print(f"   平均绝对误差: {avg_abs_error:.4f}")
            print(f"   平均相对误差: {avg_rel_error:.2f}%")
            
            return {
                'comparison_results': comparison_results,
                'summary': summary
            }
        else:
            print("❌ 没有可比较的数据")
            return None
    
    def generate_comparison_report(self, comparison_data, output_path="../../data/Adjustment_csv/7月1预测.txt"):
        """生成比较报告"""
        print("📝 正在生成比较报告...")
        
        try:
            if comparison_data is None:
                print("❌ 无比较数据可生成报告")
                return None
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=== 7月1日股票价格预测结果报告 ===\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 写入总体统计
                summary = comparison_data['summary']
                f.write("=== 总体统计 ===\n")
                f.write(f"预测股票总数: {summary['total_stocks']}\n")
                f.write(f"方向预测准确率: {summary['direction_accuracy']:.2%}\n")
                f.write(f"方向预测正确数量: {summary['correct_direction_count']}\n")
                f.write(f"平均绝对误差: {summary['avg_abs_error']:.4f}\n")
                f.write(f"平均相对误差: {summary['avg_rel_error']:.2f}%\n\n")
                
                # 写入详细结果
                f.write("=== 详细预测结果 ===\n")
                f.write("股票代码\t预测收盘价\t实际收盘价\t绝对误差\t相对误差\t方向预测\t实际方向\t方向正确\n")
                
                for result in comparison_data['comparison_results']:
                    f.write(f"{result['stock_code']}\t"
                           f"{result['predicted_close']:.4f}\t"
                           f"{result['actual_close']:.4f}\t"
                           f"{result['abs_error']:.4f}\t"
                           f"{result['rel_error']:.2f}%\t"
                           f"{result['predicted_direction']}\t"
                           f"{result['actual_direction']}\t"
                           f"{'✓' if result['direction_correct'] else '✗'}\n")
                
                # 写入模型信息
                f.write("\n=== 模型信息 ===\n")
                f.write(f"模型路径: {self.model_path}\n")
                if self.config is not None:
                    f.write(f"模型类型: {self.config.get('model_type', 'basic')}\n")
                    f.write(f"序列长度: {self.config['seq_length']}\n")
                    f.write(f"输入维度: {self.config['input_dim']}\n")
                    f.write(f"模型维度: {self.config['d_model']}\n")
                    f.write(f"注意力头数: {self.config['nhead']}\n")
                    f.write(f"层数: {self.config['num_layers']}\n")
                    f.write(f"Dropout: {self.config['dropout']}\n")
                else:
                    f.write("模型配置: 未加载\n")
                
                # 写入性能评估
                f.write("\n=== 性能评估 ===\n")
                if summary['avg_rel_error'] < 5:
                    f.write("预测精度: 优秀 (相对误差 < 5%)\n")
                elif summary['avg_rel_error'] < 10:
                    f.write("预测精度: 良好 (相对误差 < 10%)\n")
                elif summary['avg_rel_error'] < 20:
                    f.write("预测精度: 一般 (相对误差 < 20%)\n")
                else:
                    f.write("预测精度: 需要改进 (相对误差 >= 20%)\n")
                
                if summary['direction_accuracy'] > 0.6:
                    f.write("方向预测: 良好 (准确率 > 60%)\n")
                elif summary['direction_accuracy'] > 0.5:
                    f.write("方向预测: 一般 (准确率 > 50%)\n")
                else:
                    f.write("方向预测: 需要改进 (准确率 <= 50%)\n")
            
            print(f"✅ 报告已生成: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 生成报告时出错: {e}")
            return None
    
    def run_july_first_prediction(self, 
                                 data_path="../../data/test_csv",
                                 actual_data_path="../../data/Adjustment_csv/20250701_daily.csv"):
        """运行完整的7月1号预测流程"""
        print("🚀 开始运行7月1号预测流程...")
        
        try:
            # 1. 预测所有股票7月1号的价格
            predictions = self.predict_all_stocks_july_first(data_path)
            if predictions is None:
                print("❌ 预测失败")
                return False
            
            # 2. 保存预测结果到CSV
            prediction_csv_path = self.save_july_first_predictions_to_csv(predictions)
            if prediction_csv_path is None:
                print("❌ 保存预测结果失败")
                return False
            
            # 3. 加载实际数据
            actual_data = self.load_actual_july_first_data(actual_data_path)
            
            # 4. 比较预测结果与实际数据
            comparison_data = self.compare_predictions_with_actual(predictions, actual_data)
            
            # 5. 生成比较报告
            report_path = self.generate_comparison_report(comparison_data)
            
            if report_path:
                print("✅ 7月1号预测流程完成!")
                print(f"📄 预测数据文件: {prediction_csv_path}")
                print(f"📄 对比报告文件: {report_path}")
                return True
            else:
                print("⚠️  预测完成但报告生成失败")
                return False
                
        except Exception as e:
            print(f"❌ 运行预测流程时出错: {e}")
            return False

def main():
    """主程序入口"""
    print("=" * 60)
    print("🚀 股票价格预测程序启动")
    print("=" * 60)
    
    try:
        # 设置路径
        model_path = "../../models/best_model.pth"
        data_path = "../../data/test_csv"
        actual_data_path = "../../data/Adjustment_csv/20250701_daily.csv"
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("请确保模型文件已训练并保存在正确位置")
            return False
        
        # 检查测试数据目录是否存在
        if not os.path.exists(data_path):
            print(f"❌ 测试数据目录不存在: {data_path}")
            print("请确保测试数据目录存在且包含股票数据文件")
            return False
        
        # 创建预测器
        print("🔧 正在初始化预测器...")
        predictor = StockPredictor(model_path=model_path)
        
        # 运行7月1号预测流程
        success = predictor.run_july_first_prediction(
            data_path=data_path,
            actual_data_path=actual_data_path
        )
        
        if success:
            print("\n" + "=" * 60)
            print("✅ 预测程序执行成功!")
            print("📁 输出文件:")
            print("   - 预测数据: ../../data/Adjustment_csv/7月1预测.csv")
            print("   - 对比报告: ../../data/Adjustment_csv/7月1预测.txt")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ 预测程序执行失败!")
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 运行主程序
    success = main()
    
    if success:
        print("\n🎉 程序执行完成，请查看输出文件!")
    else:
        print("\n💡 程序执行失败，请检查以下几点:")
        print("   1. 模型文件是否存在并且路径正确")
        print("   2. 测试数据目录是否存在并包含CSV文件")
        print("   3. 数据格式是否正确(包含open,high,low,close,volume列)")
        print("   4. 模型配置文件是否匹配")
        
    input("\n按任意键退出...")