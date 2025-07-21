"""
股票价格预测程序 - 改进版多任务学习模型预测器
==========================================

功能：
1. 加载改进版多任务学习模型（价格、方向、幅度预测）
2. 对新的股票数据进行多任务预测
3. 可视化预测结果
4. 计算多维度评估指标
5. 支持批量预测和单股票预测

适配模型：
- ImprovedStockTransformer (多任务学习)
- MultiTaskLoss (多任务损失函数)
- 高级特征工程

作者：AI Assistant
创建时间：2024年
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入改进版模型和处理器
from learn.train.first import ImprovedStockTransformer, ImprovedStockDataProcessor, MultiTaskLoss

class ImprovedStockPredictor:
    """改进版股票价格预测器 - 支持多任务学习"""
    
    def __init__(self, model_path=None):
        """初始化预测器"""
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models/best_improved_model.pth'))
        self.model_path = model_path
        self.config_path = self.model_path.replace('.pth', '_config.pkl')
        self.processor_path = self.model_path.replace('.pth', '_processor.pkl')
        
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
                    'seq_length': 60,
                    'd_model': 256,
                    'nhead': 8,
                    'num_layers': 4,
                    'dropout': 0.15,
                    'prediction_days': 7,
                    'batch_size': 64,
                    'learning_rate': 5e-4,
                    'epochs': 150,
                    'weight_decay': 1e-5
                }
            # 加载数据处理器
            print("🔧 正在加载数据处理器...")
            if os.path.exists(self.processor_path):
                with open(self.processor_path, 'rb') as f:
                    self.processor = pickle.load(f)
                print("✅ 数据处理器加载成功")
            else:
                print("⚠️  数据处理器不存在，创建新的处理器")
                self.processor = ImprovedStockDataProcessor(
                    seq_length=self.config['seq_length'],
                    prediction_days=self.config['prediction_days']
                )
            # 创建模型
            print("🏗️  正在创建模型...")
            if self.config is None:
                print("❌ 配置为空，无法创建模型")
                return
            # ===== 关键修正：input_dim严格用processor.feature_columns长度 =====
            input_dim = len(self.processor.feature_columns) if self.processor and hasattr(self.processor, 'feature_columns') and self.processor.feature_columns else self.config.get('input_dim', 50)
            self.model = ImprovedStockTransformer(
                input_dim=input_dim,
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                num_layers=self.config['num_layers'],
                seq_len=self.config['seq_length'],
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

    def prepare_prediction_data(self, stock_data):
        """准备预测数据"""
        try:
            if self.config is None or self.processor is None:
                print("❌ 配置或数据处理器未加载，无法进行预测")
                return None
            # ===== 新增：自动兼容 vol 列名 =====
            if 'vol' not in stock_data.columns:
                for alt in ['volume', 'Volume', '成交量']:
                    if alt in stock_data.columns:
                        stock_data['vol'] = stock_data[alt]
                        print(f"⚠️ 自动将 {alt} 列重命名为 vol")
                        break
                else:
                    print(f'⚠️ 股票数据缺少成交量(vol)列，已用0填充，预测结果可能不准')
                    stock_data['vol'] = 0
            # 检查数据是否足够
            if len(stock_data) < self.config['seq_length']:
                print(f"❌ 数据不足，需要至少 {self.config['seq_length']} 条记录")
                return None
            # 添加高级特征
            print("🔧 正在添加高级特征...")
            data_with_features = self.processor.add_advanced_features(stock_data.copy())
            # ===== 关键修正：严格用processor.feature_columns和scaler处理特征 =====
            feature_cols = self.processor.feature_columns
            if not feature_cols:
                print("❌ 处理器未保存特征列，无法预测")
                return None
            # 只保留训练时的特征列
            data_features = data_with_features[feature_cols].copy()
            # 丢弃NaN
            data_features = data_features.dropna()
            if data_features.empty:
                print("❌ 处理后无有效特征数据")
                return None
            # 标准化
            features_scaled = self.processor.feature_scaler.transform(data_features)
            # 构造序列
            X = []
            for i in range(len(features_scaled) - self.config['seq_length'] + 1):
                X.append(features_scaled[i:i + self.config['seq_length']])
            X = np.array(X)
            if X is None or len(X) == 0:
                print("❌ 序列数据准备失败")
                return None
            print(f"✅ 数据准备完成: X.shape={X.shape}")
            return X
        except Exception as e:
            print(f"❌ 准备预测数据时出错: {e}")
            return None
    
    def predict_single_stock(self, stock_data, stock_code):
        """预测单只股票"""
        try:
            # 检查模型是否存在
            if self.model is None:
                print(f"❌ 模型为空，无法预测股票 {stock_code}")
                return None
            
            # 准备数据
            X = self.prepare_prediction_data(stock_data)
            if X is None:
                return None
            
            # 使用最后一个序列进行预测
            last_sequence = X[-1]
            
            # 进行多任务预测
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
                price_pred, direction_pred, magnitude_pred = self.model(X_tensor)
                
                # 获取预测结果
                predicted_price = price_pred.cpu().numpy().flatten()[0]
                predicted_direction = torch.softmax(direction_pred, dim=1).cpu().numpy()[0]
                predicted_magnitude = torch.softmax(magnitude_pred, dim=1).cpu().numpy()[0]
            
            # 获取最后一天的实际数据
            last_day_data = stock_data.iloc[-1]
            
            # 构造预测结果
            result = {
                'stock_code': stock_code,
                'last_close': last_day_data['close'],
                'predicted_price': predicted_price,
                'direction_probabilities': predicted_direction,
                'predicted_direction': np.argmax(predicted_direction),  # 0:下跌, 1:上涨
                'direction_confidence': np.max(predicted_direction),
                'magnitude_probabilities': predicted_magnitude,
                'predicted_magnitude': np.argmax(predicted_magnitude),  # 0:下跌, 1:横盘, 2:上涨
                'magnitude_confidence': np.max(predicted_magnitude),
                'last_trading_day': last_day_data.get('tradingday', 'unknown'),
                'data_points': len(stock_data),
                'true_direction': int(last_day_data['direction']) if 'direction' in last_day_data else None
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 预测股票 {stock_code} 时出错: {e}")
            return None
    
    def load_stock_data_from_csv(self, data_path="../../data/test_csv"):
        """从CSV文件加载股票数据"""
        print(f"📂 正在从CSV文件加载股票数据: {data_path}")
        
        try:
            if self.config is None:
                print("❌ 配置未加载，无法加载股票数据")
                return None
            if not os.path.exists(data_path):
                print(f"❌ 数据路径不存在: {data_path}")
                return None
            
            # 获取所有CSV文件
            csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            csv_files.sort()
            
            print(f"📅 找到 {len(csv_files)} 个CSV文件")
            
            # 用于存储每只股票的数据
            stock_data = {}
            
            # 遍历每个CSV文件
            for filename in csv_files:
                file_path = os.path.join(data_path, filename)
                
                try:
                    # 读取数据
                    daily_data = pd.read_csv(file_path)
                    
                    # 检查必要的列
                    required_cols = ['secucode', 'close', 'open', 'high', 'low', 'vol']
                    if not all(col in daily_data.columns for col in required_cols):
                        missing_cols = [col for col in required_cols if col not in daily_data.columns]
                        print(f"⚠️  跳过文件 {filename}: 缺少必要列 {missing_cols}")
                        continue
                    
                    # 重命名列以匹配期望格式
                    column_mapping = {
                        'vol': 'volume',
                        'tradingday': 'date'
                    }
                    daily_data = daily_data.rename(columns=column_mapping)
                    
                    # 为每只股票添加数据
                    for _, row in daily_data.iterrows():
                        stock_code = row['secucode']
                        
                        if stock_code not in stock_data:
                            stock_data[stock_code] = []
                        
                        # 添加数据
                        stock_data[stock_code].append({
                            'date': row.get('date', filename.replace('.csv', '')),
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
            processed_data = {}
            for stock_code, history in stock_data.items():
                if self.config is None:
                    print("❌ 配置未加载，无法判断序列长度")
                    continue
                if len(history) >= self.config['seq_length']:
                    df = pd.DataFrame(history)
                    df = df.sort_values('date')
                    processed_data[stock_code] = df
                    print(f"✅ 股票 {stock_code}: {len(df)} 个交易日数据")
            
            print(f"✅ 成功加载 {len(processed_data)} 只股票的数据")
            return processed_data
            
        except Exception as e:
            print(f"❌ 加载股票数据时出错: {e}")
            return None
    
    def predict_all_stocks(self, data_path="../../data/test_csv"):
        """预测所有股票"""
        print("🎯 开始预测所有股票...")
        
        # 加载股票数据
        stock_data = self.load_stock_data_from_csv(data_path)
        if stock_data is None:
            print("❌ 无法加载股票数据")
            return None
        
        # 预测每只股票
        predictions = {}
        total_stocks = len(stock_data)
        successful_predictions = 0
        
        print(f"\n📊 开始预测 {total_stocks} 只股票...")
        
        for i, (stock_code, data) in enumerate(stock_data.items(), 1):
            print(f"\n🔮 [{i}/{total_stocks}] 预测 {stock_code}...")
            
            result = self.predict_single_stock(data, stock_code)
            if result:
                predictions[stock_code] = result
                successful_predictions += 1
                
                direction_text = "上涨" if result['predicted_direction'] == 1 else "下跌"
                magnitude_text = ["下跌", "横盘", "上涨"][result['predicted_magnitude']]
                
                print(f"✅ {stock_code}: 预测价格 {result['predicted_price']:.4f}")
                print(f"   方向: {direction_text} (置信度: {result['direction_confidence']:.2%})")
                print(f"   幅度: {magnitude_text} (置信度: {result['magnitude_confidence']:.2%})")
            else:
                print(f"❌ {stock_code}: 预测失败")
        
        print(f"\n✅ 完成预测，成功预测 {successful_predictions}/{total_stocks} 只股票")
        return predictions
    
    def save_predictions_to_csv(self, predictions, output_path="../../data/Adjustment_csv/多任务预测结果.csv"):
        """保存预测结果到CSV文件"""
        print("💾 正在保存预测结果到CSV文件...")
        
        try:
            # 创建预测数据列表
            prediction_data = []
            
            for stock_code, result in predictions.items():
                direction_text = "上涨" if result['predicted_direction'] == 1 else "下跌"
                magnitude_text = ["下跌", "横盘", "上涨"][result['predicted_magnitude']]
                
                row = {
                    'stock_code': stock_code,
                    'last_close': result['last_close'],
                    'predicted_price': result['predicted_price'],
                    'predicted_direction': direction_text,
                    'direction_confidence': result['direction_confidence'],
                    'predicted_magnitude': magnitude_text,
                    'magnitude_confidence': result['magnitude_confidence'],
                    'last_trading_day': result['last_trading_day'],
                    'data_points': result['data_points']
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
    
    def generate_prediction_report(self, predictions, output_path="../../data/Adjustment_csv/多任务预测报告.txt"):
        """生成预测报告，统计方向预测准确率（与20250701_daily.csv真实direction对比）"""
        print("📝 正在生成预测报告...")
        try:
            # 读取真实标签
            real_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/Adjustment_csv/20250701_daily.csv'))
            real_df = None
            if os.path.exists(real_csv_path):
                real_df = pd.read_csv(real_csv_path)
                real_direction_dict = {str(row['secucode']): int(row['direction']) for _, row in real_df.iterrows() if 'direction' in row}
            else:
                real_direction_dict = {}
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("多任务学习股票预测报告\n")
                f.write("="*50 + "\n")
                f.write(f"总预测股票数: {len(predictions)}\n")
                # 统计方向预测准确率（与csv真实标签对比）
                y_true = []
                y_pred = []
                for stock_code, result in predictions.items():
                    code = str(stock_code)
                    if code in real_direction_dict:
                        y_true.append(real_direction_dict[code])
                        y_pred.append(result['predicted_direction'])
                if y_true and y_pred:
                    from sklearn.metrics import accuracy_score
                    acc = accuracy_score(y_true, y_pred)
                    f.write(f"与20250701_daily.csv真实标签对比的方向预测二分类准确率: {acc:.4f}\n")
                    print(f"🎯 与20250701_daily.csv真实标签对比的方向预测二分类准确率: {acc:.4f}")
                else:
                    f.write("未找到真实标签，无法统计准确率\n")
                    print("⚠️ 未找到真实标签，无法统计准确率")
                # 总体统计
                total_stocks = len(predictions)
                up_count = sum(1 for p in predictions.values() if p['predicted_direction'] == 1)
                down_count = total_stocks - up_count
                
                avg_direction_confidence = np.mean([p['direction_confidence'] for p in predictions.values()])
                avg_magnitude_confidence = np.mean([p['magnitude_confidence'] for p in predictions.values()])
                
                f.write("=== 总体统计 ===\n")
                f.write(f"预测股票总数: {total_stocks}\n")
                f.write(f"预测上涨股票数: {up_count} ({up_count/total_stocks:.1%})\n")
                f.write(f"预测下跌股票数: {down_count} ({down_count/total_stocks:.1%})\n")
                f.write(f"平均方向预测置信度: {avg_direction_confidence:.2%}\n")
                f.write(f"平均幅度预测置信度: {avg_magnitude_confidence:.2%}\n\n")
                
                # 幅度分布
                magnitude_counts = {}
                for p in predictions.values():
                    mag = p['predicted_magnitude']
                    magnitude_counts[mag] = magnitude_counts.get(mag, 0) + 1
                
                f.write("=== 幅度预测分布 ===\n")
                magnitude_names = ["下跌", "横盘", "上涨"]
                for i, name in enumerate(magnitude_names):
                    count = magnitude_counts.get(i, 0)
                    f.write(f"{name}: {count} 只 ({count/total_stocks:.1%})\n")
                f.write("\n")
                
                # 详细结果
                f.write("=== 详细预测结果 ===\n")
                f.write("股票代码\t预测价格\t方向\t方向置信度\t幅度\t幅度置信度\t数据点数\n")
                
                for stock_code, result in predictions.items():
                    direction_text = "上涨" if result['predicted_direction'] == 1 else "下跌"
                    magnitude_text = ["下跌", "横盘", "上涨"][result['predicted_magnitude']]
                    
                    f.write(f"{stock_code}\t"
                           f"{result['predicted_price']:.4f}\t"
                           f"{direction_text}\t"
                           f"{result['direction_confidence']:.2%}\t"
                           f"{magnitude_text}\t"
                           f"{result['magnitude_confidence']:.2%}\t"
                           f"{result['data_points']}\n")
                
                # 模型信息
                f.write("\n=== 模型信息 ===\n")
                f.write(f"模型路径: {self.model_path}\n")
                if self.config is not None:
                    f.write(f"序列长度: {self.config['seq_length']}\n")
                    f.write(f"模型维度: {self.config['d_model']}\n")
                    f.write(f"注意力头数: {self.config['nhead']}\n")
                    f.write(f"层数: {self.config['num_layers']}\n")
                    f.write(f"Dropout: {self.config['dropout']}\n")
                    f.write(f"预测天数: {self.config['prediction_days']}\n")
                else:
                    f.write("模型配置: 未加载\n")
                
                # 性能评估
                f.write("\n=== 性能评估 ===\n")
                if avg_direction_confidence > 0.7:
                    f.write("方向预测置信度: 高 (>70%)\n")
                elif avg_direction_confidence > 0.5:
                    f.write("方向预测置信度: 中等 (50-70%)\n")
                else:
                    f.write("方向预测置信度: 低 (<50%)\n")
                
                if avg_magnitude_confidence > 0.6:
                    f.write("幅度预测置信度: 高 (>60%)\n")
                elif avg_magnitude_confidence > 0.4:
                    f.write("幅度预测置信度: 中等 (40-60%)\n")
                else:
                    f.write("幅度预测置信度: 低 (<40%)\n")
                
                f.write("\n详细预测结果（部分）：\n")
                # 修正：支持dict或list类型的predictions
                if isinstance(predictions, dict):
                    pred_list = list(predictions.values())
                else:
                    pred_list = predictions
                for pred in pred_list[:10]:
                    f.write(str(pred) + "\n")
            
            print(f"✅ 报告已生成: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 生成报告时出错: {e}")
            return None
    
    def visualize_predictions(self, predictions, output_path="../../results/predictions/prediction_visualization.png"):
        """可视化预测结果"""
        print("📊 正在生成预测可视化...")
        
        try:
            if not predictions:
                print("❌ 无预测数据可可视化")
                return None
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('多任务学习股票预测结果可视化', fontsize=16, fontweight='bold')
            
            # 1. 方向预测分布
            directions = [p['predicted_direction'] for p in predictions.values()]
            direction_counts = [directions.count(0), directions.count(1)]
            direction_labels = ['下跌', '上涨']
            
            axes[0, 0].pie(direction_counts, labels=direction_labels, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('方向预测分布')
            
            # 2. 幅度预测分布
            magnitudes = [p['predicted_magnitude'] for p in predictions.values()]
            magnitude_counts = [magnitudes.count(0), magnitudes.count(1), magnitudes.count(2)]
            magnitude_labels = ['下跌', '横盘', '上涨']
            
            axes[0, 1].bar(magnitude_labels, magnitude_counts, color=['red', 'gray', 'green'])
            axes[0, 1].set_title('幅度预测分布')
            axes[0, 1].set_ylabel('股票数量')
            
            # 3. 置信度分布
            direction_confidences = [p['direction_confidence'] for p in predictions.values()]
            magnitude_confidences = [p['magnitude_confidence'] for p in predictions.values()]
            
            axes[1, 0].hist(direction_confidences, bins=20, alpha=0.7, label='方向预测', color='blue')
            axes[1, 0].hist(magnitude_confidences, bins=20, alpha=0.7, label='幅度预测', color='orange')
            axes[1, 0].set_title('预测置信度分布')
            axes[1, 0].set_xlabel('置信度')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].legend()
            
            # 4. 价格预测分布
            predicted_prices = [p['predicted_price'] for p in predictions.values()]
            last_prices = [p['last_close'] for p in predictions.values()]
            
            axes[1, 1].scatter(last_prices, predicted_prices, alpha=0.6)
            axes[1, 1].plot([min(last_prices), max(last_prices)], [min(last_prices), max(last_prices)], 'r--', lw=2)
            axes[1, 1].set_title('预测价格 vs 最后收盘价')
            axes[1, 1].set_xlabel('最后收盘价')
            axes[1, 1].set_ylabel('预测价格')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ 可视化已保存到: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ 生成可视化时出错: {e}")
            return None
    
    def run_prediction_pipeline(self, data_path="../../data/test_csv"):
        """运行完整的预测流程"""
        print("🚀 开始运行多任务学习预测流程...")
        
        try:
            # 1. 预测所有股票
            predictions = self.predict_all_stocks(data_path)
            if predictions is None:
                print("❌ 预测失败")
                return False
            
            # 2. 保存预测结果到CSV
            prediction_csv_path = self.save_predictions_to_csv(predictions)
            if prediction_csv_path is None:
                print("❌ 保存预测结果失败")
                return False
            
            # 3. 生成预测报告
            report_path = self.generate_prediction_report(predictions)
            
            # 4. 生成可视化
            viz_path = self.visualize_predictions(predictions)
            
            print("✅ 多任务学习预测流程完成!")
            print(f"📄 预测数据文件: {prediction_csv_path}")
            if report_path:
                print(f"📄 预测报告文件: {report_path}")
            if viz_path:
                print(f"📊 可视化文件: {viz_path}")
            return True
                
        except Exception as e:
            print(f"❌ 运行预测流程时出错: {e}")
            return False

def main():
    """主程序入口"""
    print("=" * 60)
    print("🚀 改进版多任务学习股票预测程序启动")
    print("=" * 60)
    
    try:
        # 设置路径
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models/best_improved_model.pth'))
        data_path = "../../data/test_csv"
        
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
        print("🔧 正在初始化改进版预测器...")
        predictor = ImprovedStockPredictor(model_path=model_path)
        
        # 运行预测流程
        success = predictor.run_prediction_pipeline(data_path=data_path)
        
        if success:
            print("\n" + "=" * 60)
            print("✅ 改进版预测程序执行成功!")
            print("📁 输出文件:")
            print("   - 预测数据: ../../data/Adjustment_csv/多任务预测结果.csv")
            print("   - 预测报告: ../../data/Adjustment_csv/多任务预测报告.txt")
            print("   - 可视化: ../../results/predictions/prediction_visualization.png")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ 改进版预测程序执行失败!")
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
        print("\n🎉 改进版预测程序执行完成，请查看输出文件!")
    else:
        print("\n💡 程序执行失败，请检查以下几点:")
        print("   1. 模型文件是否存在并且路径正确")
        print("   2. 测试数据目录是否存在并包含CSV文件")
        print("   3. 数据格式是否正确(包含secucode,open,high,low,close,vol列)")
        print("   4. 模型配置文件是否匹配")
        
    input("\n按任意键退出...")