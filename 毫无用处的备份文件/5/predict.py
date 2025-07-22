import pandas as pd
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 修复Ubuntu系统中文字体显示问题
import matplotlib
import platform

def setup_chinese_fonts():
    """设置中文字体，兼容不同操作系统"""
    system = platform.system()
    
    if system == "Linux":  # Ubuntu等Linux系统
        # 尝试多种可能的中文字体
        font_candidates = [
            'WenQuanYi Micro Hei',      # 文泉驿微米黑
            'WenQuanYi Zen Hei',        # 文泉驿正黑
            'Noto Sans CJK SC',         # 思源黑体
            'Source Han Sans CN',       # 思源黑体
            'DejaVu Sans',              # 备选方案
        ]
        
        # 检查系统可用字体
        available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        
        # 选择第一个可用的中文字体
        selected_font = 'DejaVu Sans'  # 默认字体
        for font in font_candidates:
            if font in available_fonts:
                selected_font = font
                break
        
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
        print(f"使用字体: {selected_font}")
        
    elif system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    elif system == "Darwin":  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 如果还是显示不了中文，给出安装提示
    try:
        # 测试中文显示
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
        plt.close(fig)
    except:
        print("警告: 中文字体可能无法正常显示")
        print("Ubuntu系统建议安装中文字体:")
        print("sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei")
        print("或者: sudo apt-get install fonts-noto-cjk")

# 初始化字体设置
setup_chinese_fonts()

class StockPredictor:
    def __init__(self):
        # 设置路径（使用相对路径）
        self.model_path = "./models2"
        self.test_data_path = "./data/test_csv2"
        self.real_data_path = "./data/Adjustment_csv2"
        self.results_path = "./results"
        
        # 创建结果保存目录
        os.makedirs(self.results_path, exist_ok=True)
        
        # 模型参数
        self.sequence_length = 60
        self.features = ['open', 'high', 'low', 'close', 'vol', 'amount']
        self.target = 'close'
        
        # 预测日期范围
        self.pred_start_date = '2025-06-03'
        self.pred_end_date = '2025-07-01'

    def get_trading_days(self, start_date, end_date):
        """生成交易日列表（排除周末）"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        trading_days = []
        current_date = start
        while current_date <= end:
            # 排除周末（0=周一, 6=周日）
            if current_date.weekday() < 5:
                trading_days.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
        return trading_days

    def build_model_struct(self, input_shape):
        """重建模型结构（与训练时保持一致）"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def load_model_and_scalers(self, stock_code):
        """加载模型和缩放器"""
        try:
            model_file = f"{self.model_path}/{stock_code}_model.h5"
            best_model_file = f"{self.model_path}/{stock_code}_best.h5"
            scaler_file = f"{self.model_path}/{stock_code}_scalers.pkl"
            
            if not os.path.exists(scaler_file):
                return None, None
            
            # 首先尝试直接加载模型
            model = None
            for model_path in [best_model_file, model_file]:
                if os.path.exists(model_path):
                    try:
                        # 方法1: 使用custom_objects解决函数识别问题
                        model = tf.keras.models.load_model(
                            model_path,
                            custom_objects={
                                'mse': tf.keras.losses.MeanSquaredError(),
                                'mae': tf.keras.metrics.MeanAbsoluteError()
                            }
                        )
                        break
                    except Exception as load_error1:
                        try:
                            # 方法2: 重建模型结构并加载权重
                            print(f"尝试重建模型结构加载权重: {stock_code}")
                            model = self.build_model_struct((self.sequence_length, len(self.features)))
                            model.load_weights(model_path)
                            break
                        except Exception as load_error2:
                            print(f"加载 {model_path} 失败: {str(load_error2)}")
                            continue
            
            if model is None:
                return None, None
                
            with open(scaler_file, 'rb') as f:
                feature_scaler, target_scaler = pickle.load(f)
                
            return model, (feature_scaler, target_scaler)
            
        except Exception as e:
            print(f"加载模型 {stock_code} 时出错: {str(e)}")
            return None, None

    def load_test_data(self, stock_code):
        """加载测试数据"""
        try:
            file_path = os.path.join(self.test_data_path, f"{stock_code}.csv")
            if not os.path.exists(file_path):
                return None
            
            df = pd.read_csv(file_path)
            df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y%m%d')
            df = df.sort_values('tradingday').reset_index(drop=True)
            
            return df
        except Exception as e:
            print(f"加载测试数据 {stock_code} 时出错: {str(e)}")
            return None

    def load_real_data(self, stock_code):
        """加载真实数据"""
        try:
            file_path = os.path.join(self.real_data_path, f"{stock_code}.csv")
            if not os.path.exists(file_path):
                return None
            
            df = pd.read_csv(file_path)
            df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y%m%d')
            df = df.sort_values('tradingday').reset_index(drop=True)
            
            # 筛选预测期间的数据
            start_date = datetime.strptime(self.pred_start_date, '%Y-%m-%d')
            end_date = datetime.strptime(self.pred_end_date, '%Y-%m-%d')
            df_filtered = df[(df['tradingday'] >= start_date) & (df['tradingday'] <= end_date)]
            
            return df_filtered
        except Exception as e:
            print(f"加载真实数据 {stock_code} 时出错: {str(e)}")
            return None

    def predict_future_prices(self, model, test_data, scalers, prediction_days):
        """预测未来股价"""
        feature_scaler, target_scaler = scalers
        
        # 获取最后sequence_length天的数据作为基础输入
        recent_data = test_data[self.features].tail(self.sequence_length).values
        recent_scaled = feature_scaler.transform(recent_data)
        
        predictions = []
        current_sequence = recent_scaled.copy()
        
        for _ in range(prediction_days):
            # 准备预测输入
            X_pred = current_sequence.reshape(1, self.sequence_length, len(self.features))
            
            # 预测下一天收盘价
            pred_scaled = model.predict(X_pred, verbose=0)
            pred_price = target_scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred_price)
            
            # 更新序列（使用预测的收盘价作为下一天的数据）
            # 假设其他特征与收盘价有相关性
            last_row = current_sequence[-1].copy()
            
            # 简单假设：开盘价=前一天收盘价，其他价格围绕收盘价波动
            pred_scaled_price = pred_scaled[0][0]
            
            # 更新最后一行的收盘值
            last_row[3] = pred_scaled_price  # close在features中的位置
            
            # 滑动窗口：移除第一天，添加新预测的一天
            current_sequence = np.vstack([current_sequence[1:], last_row])
            
        return predictions

    def generate_prediction_data(self, stock_code, predictions, trading_days, last_real_data):
        """生成完整的预测数据"""
        pred_data = []
        
        for i, (day, close_price) in enumerate(zip(trading_days, predictions)):
            # 简单的价格生成策略
            if i == 0 and last_real_data is not None:
                # 第一天使用最后一天真实数据作为参考
                open_price = last_real_data['close']
            else:
                # 后续天数的开盘价等于前一天收盘价
                open_price = predictions[i-1] if i > 0 else close_price
            
            # 生成高低价（假设波动范围在±2%内）
            volatility = 0.02
            high_price = close_price * (1 + np.random.uniform(0, volatility))
            low_price = close_price * (1 - np.random.uniform(0, volatility))
            
            # 确保价格逻辑正确
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            pred_data.append({
                'tradingday': day,
                'secucode': stock_code,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'vol': 0,  # 成交量预测较复杂，这里设置为0
                'amount': 0  # 成交额预测较复杂，这里设置为0
            })
        
        return pd.DataFrame(pred_data)

    def plot_comparison(self, stock_code, pred_data, real_data, metrics):
        """绘制预测与真实数据对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{stock_code} 股票预测 vs 真实数据对比', fontsize=16, fontweight='bold')
        
        # 转换日期格式
        pred_data['date'] = pd.to_datetime(pred_data['tradingday'], format='%Y%m%d')
        real_data['date'] = real_data['tradingday']
        
        # 1. 收盘价对比
        ax1.plot(pred_data['date'], pred_data['close'], 'b-', label='预测价格', linewidth=2)
        ax1.plot(real_data['date'], real_data['close'], 'r-', label='真实价格', linewidth=2)
        ax1.set_title('收盘价对比')
        ax1.set_ylabel('价格(元)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 开盘价对比
        ax2.plot(pred_data['date'], pred_data['open'], 'b-', label='预测开盘', linewidth=2)
        ax2.plot(real_data['date'], real_data['open'], 'r-', label='真实开盘', linewidth=2)
        ax2.set_title('开盘价对比')
        ax2.set_ylabel('价格(元)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 高低价对比
        ax3.plot(pred_data['date'], pred_data['high'], 'b-', label='预测最高', alpha=0.7)
        ax3.plot(pred_data['date'], pred_data['low'], 'b--', label='预测最低', alpha=0.7)
        ax3.plot(real_data['date'], real_data['high'], 'r-', label='真实最高', alpha=0.7)
        ax3.plot(real_data['date'], real_data['low'], 'r--', label='真实最低', alpha=0.7)
        ax3.fill_between(pred_data['date'], pred_data['low'], pred_data['high'], 
                        alpha=0.2, color='blue', label='预测价格区间')
        ax3.fill_between(real_data['date'], real_data['low'], real_data['high'], 
                        alpha=0.2, color='red', label='真实价格区间')
        ax3.set_title('最高价和最低价对比')
        ax3.set_ylabel('价格(元)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 预测误差分析
        if len(pred_data) == len(real_data):
            errors = np.abs(pred_data['close'].values - real_data['close'].values)
            ax4.bar(range(len(errors)), errors, alpha=0.7, color='orange')
            ax4.set_title('每日预测绝对误差')
            ax4.set_xlabel('交易日')
            ax4.set_ylabel('绝对误差(元)')
            ax4.grid(True, alpha=0.3)
            
            # 添加平均误差线
            mean_error = np.mean(errors)
            ax4.axhline(y=mean_error, color='red', linestyle='--', 
                       label=f'平均误差: {mean_error:.2f}')
            ax4.legend()
        
        # 添加评估指标文本
        textstr = f'''评估指标:
MSE: {metrics['mse']:.6f}
RMSE: {metrics['rmse']:.4f}
MAE: {metrics['mae']:.4f}
MAPE: {metrics['mape']:.2f}%
R²: {metrics['r2']:.4f}'''
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        fig.text(0.02, 0.02, textstr, fontsize=10, bbox=props)
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(f"{self.results_path}/{stock_code}_prediction_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存 {stock_code} 的对比图")

    def calculate_metrics(self, pred_values, real_values):
        """计算评估指标"""
        mse = mean_squared_error(real_values, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(real_values, pred_values)
        
        # MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((real_values - pred_values) / real_values)) * 100
        
        # R²决定系数
        r2 = r2_score(real_values, pred_values)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }

    def predict_single_stock(self, stock_code):
        """预测单只股票"""
        print(f"开始预测股票 {stock_code}...")
        
        # 加载模型和缩放器
        model, scalers = self.load_model_and_scalers(stock_code)
        if model is None:
            print(f"无法加载股票 {stock_code} 的模型")
            return None
        
        # 加载测试数据
        test_data = self.load_test_data(stock_code)
        if test_data is None:
            print(f"无法加载股票 {stock_code} 的测试数据")
            return None
        
        # 加载真实数据
        real_data = self.load_real_data(stock_code)
        if real_data is None:
            print(f"无法加载股票 {stock_code} 的真实数据")
            return None
        
        # 生成交易日列表
        trading_days = self.get_trading_days(self.pred_start_date, self.pred_end_date)
        
        # 进行预测
        predictions = self.predict_future_prices(model, test_data, scalers, len(trading_days))
        
        # 生成预测数据DataFrame
        last_real_data = test_data.iloc[-1] if not test_data.empty else None
        pred_data = self.generate_prediction_data(stock_code, predictions, trading_days, last_real_data)
        
        # 确保真实数据和预测数据的日期匹配
        real_data_filtered = real_data[real_data['tradingday'].dt.strftime('%Y%m%d').isin(trading_days)]
        
        if len(real_data_filtered) == 0:
            print(f"股票 {stock_code} 没有对应时间段的真实数据")
            return None
        
        # 计算评估指标
        if len(pred_data) == len(real_data_filtered):
            metrics = self.calculate_metrics(
                pred_data['close'].values,
                real_data_filtered['close'].values
            )
        else:
            print(f"Warning: {stock_code} 预测数据和真实数据长度不匹配")
            # 取较短的长度进行计算
            min_len = min(len(pred_data), len(real_data_filtered))
            metrics = self.calculate_metrics(
                pred_data['close'].head(min_len).values,
                real_data_filtered['close'].head(min_len).values
            )
        
        # 绘制对比图
        self.plot_comparison(stock_code, pred_data, real_data_filtered, metrics)
        
        return {
            'stock_code': stock_code,
            'metrics': metrics,
            'pred_data': pred_data,
            'real_data': real_data_filtered,
            'prediction_days': len(trading_days)
        }

    def run_all_predictions(self):
        """运行所有股票的预测"""
        # 获取所有已训练的模型
        model_files = glob.glob(os.path.join(self.model_path, "*_model.h5"))
        stock_codes = [os.path.basename(f).replace('_model.h5', '') for f in model_files]
        
        if not stock_codes:
            print("未找到训练好的模型")
            return
        
        print(f"找到 {len(stock_codes)} 个训练好的模型")
        
        results = []
        successful_predictions = 0
        failed_predictions = 0
        
        for i, stock_code in enumerate(stock_codes, 1):
            print(f"\n进度: {i}/{len(stock_codes)}")
            result = self.predict_single_stock(stock_code)
            
            if result:
                results.append(result)
                successful_predictions += 1
            else:
                failed_predictions += 1
        
        # 生成汇总报告
        self.generate_summary_report(results, successful_predictions, failed_predictions)
        
        print(f"\n预测完成!")
        print(f"成功预测: {successful_predictions} 只股票")
        print(f"预测失败: {failed_predictions} 只股票")
        print(f"结果保存路径: {self.results_path}")

    def generate_summary_report(self, results, success, failed):
        """生成预测报告"""
        report_path = os.path.join(self.results_path, "prediction_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("股票价格预测模型评估报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"预测时间段: {self.pred_start_date} 至 {self.pred_end_date}\n\n")
            
            # 整体统计
            f.write("整体统计:\n")
            f.write("-" * 30 + "\n")
            f.write(f"总股票数量: {success + failed}\n")
            f.write(f"成功预测: {success}\n")
            f.write(f"预测失败: {failed}\n")
            f.write(f"成功率: {success/(success + failed)*100:.1f}%\n\n")
            
            if results:
                # 计算整体指标
                all_mse = [r['metrics']['mse'] for r in results]
                all_rmse = [r['metrics']['rmse'] for r in results]
                all_mae = [r['metrics']['mae'] for r in results]
                all_mape = [r['metrics']['mape'] for r in results]
                all_r2 = [r['metrics']['r2'] for r in results]
                
                f.write("整体性能指标:\n")
                f.write("-" * 30 + "\n")
                f.write(f"平均MSE: {np.mean(all_mse):.6f} (±{np.std(all_mse):.6f})\n")
                f.write(f"平均RMSE: {np.mean(all_rmse):.4f} (±{np.std(all_rmse):.4f})\n")
                f.write(f"平均MAE: {np.mean(all_mae):.4f} (±{np.std(all_mae):.4f})\n")
                f.write(f"平均MAPE: {np.mean(all_mape):.2f}% (±{np.std(all_mape):.2f}%)\n")
                f.write(f"平均R²: {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})\n\n")
                
                # 性能分级
                excellent = sum(1 for r in results if r['metrics']['mape'] < 5)
                good = sum(1 for r in results if 5 <= r['metrics']['mape'] < 10)
                fair = sum(1 for r in results if 10 <= r['metrics']['mape'] < 20)
                poor = sum(1 for r in results if r['metrics']['mape'] >= 20)
                
                f.write("预测精度分级（基于MAPE）:\n")
                f.write("-" * 30 + "\n")
                f.write(f"优秀 (<5%): {excellent} 只股票 ({excellent/len(results)*100:.1f}%)\n")
                f.write(f"良好 (5-10%): {good} 只股票 ({good/len(results)*100:.1f}%)\n")
                f.write(f"一般 (10-20%): {fair} 只股票 ({fair/len(results)*100:.1f}%)\n")
                f.write(f"较差 (>20%): {poor} 只股票 ({poor/len(results)*100:.1f}%)\n\n")
                
                # 最佳和最差表现
                best_stock = min(results, key=lambda x: x['metrics']['mape'])
                worst_stock = max(results, key=lambda x: x['metrics']['mape'])
                
                f.write("最佳预测表现:\n")
                f.write("-" * 30 + "\n")
                f.write(f"股票代码: {best_stock['stock_code']}\n")
                f.write(f"MAPE: {best_stock['metrics']['mape']:.2f}%\n")
                f.write(f"RMSE: {best_stock['metrics']['rmse']:.4f}\n")
                f.write(f"R²: {best_stock['metrics']['r2']:.4f}\n\n")
                
                f.write("最差预测表现:\n")
                f.write("-" * 30 + "\n")
                f.write(f"股票代码: {worst_stock['stock_code']}\n")
                f.write(f"MAPE: {worst_stock['metrics']['mape']:.2f}%\n")
                f.write(f"RMSE: {worst_stock['metrics']['rmse']:.4f}\n")
                f.write(f"R²: {worst_stock['metrics']['r2']:.4f}\n\n")
                
                # 详细结果
                f.write("详细预测结果:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'股票代码':<10} {'MSE':<12} {'RMSE':<8} {'MAE':<8} {'MAPE(%)':<10} {'R²':<8}\n")
                f.write("-" * 80 + "\n")
                
                for result in sorted(results, key=lambda x: x['metrics']['mape']):
                    metrics = result['metrics']
                    f.write(f"{result['stock_code']:<10} {metrics['mse']:<12.6f} "
                           f"{metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} "
                           f"{metrics['mape']:<10.2f} {metrics['r2']:<8.4f}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("报告说明:\n")
                f.write("MSE: 均方误差，越小越好\n")
                f.write("RMSE: 均方根误差，越小越好\n")
                f.write("MAE: 平均绝对误差，越小越好\n")
                f.write("MAPE: 平均绝对百分比误差，越小越好\n")
                f.write("R²: 决定系数，越接近1越好\n")
        
        print(f"预测报告已保存到: {report_path}")

def main():
    """主函数"""    
    predictor = StockPredictor()
    print("开始股票预测和评估...")
    print(f"预测时间段: {predictor.pred_start_date} 至 {predictor.pred_end_date}")
    predictor.run_all_predictions()

if __name__ == "__main__":
    main()