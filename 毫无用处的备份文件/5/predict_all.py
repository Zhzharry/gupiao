import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, model_path, test_data_path, actual_data_path, results_path):
        """
        Initialize Stock Predictor
        
        Args:
            model_path: Path to saved model files
            test_data_path: Path to test CSV files
            actual_data_path: Path to actual data for comparison
            results_path: Path to save results
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.actual_data_path = actual_data_path
        self.results_path = results_path
        
        # Create results directory
        os.makedirs(self.results_path, exist_ok=True)
        
        # Load model and scalers
        self.load_model_components()
        
    def load_model_components(self):
        """Load trained model and preprocessing components"""
        print("Loading model components...")
        
        try:
            # 检查模型目录是否存在
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            
            # 定义可能的模型文件名
            possible_model_files = [
                'stock_prediction_model.h5',
                'model.h5',
                'best_model.h5'
            ]
            
            model_file = None
            # 检查可能的模型文件是否存在
            for filename in possible_model_files:
                filepath = os.path.join(self.model_path, filename)
                if os.path.exists(filepath):
                    model_file = filepath
                    break
            
            if model_file is None:
                available_files = os.listdir(self.model_path)
                raise FileNotFoundError(
                    f"No model file found in {self.model_path}. "
                    f"Available files: {available_files}"
                )
            
            print(f"Found model file: {model_file}")
            
            # 加载模型
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mean_squared_error': tf.keras.losses.MeanSquaredError()
            }
            
            self.model = tf.keras.models.load_model(
                model_file,
                custom_objects=custom_objects,
                compile=True
            )
            print(f"✅ Model loaded successfully from: {model_file}")
            
            # 加载scaler文件
            scaler_files = {
                'feature_scaler': 'feature_scaler.pkl',
                'target_scaler': 'target_scaler.pkl',
                'model_config': 'model_config.pkl'
            }
            
            for name, filename in scaler_files.items():
                filepath = os.path.join(self.model_path, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"{name} file not found: {filepath}")
                
                with open(filepath, 'rb') as f:
                    if name == 'feature_scaler':
                        self.feature_scaler = pickle.load(f)
                    elif name == 'target_scaler':
                        self.target_scaler = pickle.load(f)
                    elif name == 'model_config':
                        self.config = pickle.load(f)
                
                print(f"✅ {name} loaded: {filepath}")
            
            # 检查必要的配置参数
            required_configs = ['sequence_length', 'prediction_days']
            for config in required_configs:
                if config not in self.config:
                    raise ValueError(f"Missing required config: {config}")
            
            self.sequence_length = self.config['sequence_length']
            self.prediction_days = self.config['prediction_days']
            
            print(f"✅ Configuration loaded: {self.sequence_length} days → {self.prediction_days} days average")
            
        except Exception as e:
            print(f"❌ Error loading model components: {str(e)}")
            raise e
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators (same as training)"""
        try:
            # Moving averages
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['ma50'] = df['close'].rolling(window=50, min_periods=1).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Price changes
            df['price_change'] = df['close'].pct_change().fillna(0)
            df['price_change_3d'] = df['close'].pct_change(periods=3).fillna(0)
            df['price_change_5d'] = df['close'].pct_change(periods=5).fillna(0)
            df['volume_change'] = df['vol'].pct_change().fillna(0)
            
            # MACD
            exp1 = df['close'].ewm(span=12, min_periods=1).mean()
            exp2 = df['close'].ewm(span=26, min_periods=1).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
            bb_std = df['close'].rolling(window=20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Bollinger Band position
            bb_range = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = np.where(bb_range > 0, 
                                    (df['close'] - df['bb_lower']) / bb_range,
                                    0.5)
            
            # Volatility indicators
            df['volatility'] = df['close'].rolling(window=20, min_periods=1).std()
            df['high_low_ratio'] = df['high'] / df['low']
            
            # Volume indicators
            df['volume_ma'] = df['vol'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['vol'] / (df['volume_ma'] + 1e-10)
            
            # Clean infinite values - 修改了这部分代码
            df = df.replace([np.inf, -np.inf], np.nan)
            # 使用ffill和bfill的字符串形式
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            raise e
    
    def predict_future_prices(self, stock_code, test_data, prediction_days=29):
        """
        Predict future stock prices for next month
        
        Args:
            stock_code: Stock code
            test_data: Historical data for prediction
            prediction_days: Number of days to predict (29 days for month)
            
        Returns:
            predicted_prices: Array of predicted prices
            prediction_dates: Array of prediction dates
        """
        try:
            # Prepare features
            feature_columns = ['open', 'high', 'low', 'close', 'vol', 'amount', 'deals',
                              'ma5', 'ma10', 'ma20', 'ma50', 'rsi', 'price_change', 
                              'price_change_3d', 'price_change_5d', 'volume_change',
                              'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                              'volatility', 'high_low_ratio', 'volume_ratio']
            
            available_features = [col for col in feature_columns if col in test_data.columns]
            
            if len(test_data) < self.sequence_length:
                raise ValueError(f"Insufficient data: need {self.sequence_length} days, got {len(test_data)}")
            
            # Get the last sequence for prediction
            features = test_data[available_features].values
            features_scaled = self.feature_scaler.transform(features)
            
            # Use the last sequence_length days as input
            last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Predict average price for next prediction_days
            predicted_scaled = self.model.predict(last_sequence, verbose=0)[0][0]
            predicted_avg_price = self.target_scaler.inverse_transform([[predicted_scaled]])[0][0]
            
            # Generate prediction dates (20250603-20250701)
            last_date = pd.to_datetime('20250530')
            prediction_dates = []
            current_date = last_date + timedelta(days=1)
            
            for i in range(prediction_days):
                # Skip weekends
                while current_date.weekday() >= 5:  # Saturday=5, Sunday=6
                    current_date += timedelta(days=1)
                prediction_dates.append(current_date)
                current_date += timedelta(days=1)
            
            # Generate daily prices around the predicted average
            # Use last known price as base and add some realistic variation
            last_close = test_data['close'].iloc[-1]
            price_trend = (predicted_avg_price - last_close) / len(prediction_dates)
            
            predicted_prices = []
            for i, date in enumerate(prediction_dates):
                # Base price with trend
                base_price = last_close + (price_trend * (i + 1))
                
                # Add small random variation (±2%)
                np.random.seed(hash(f"{stock_code}_{date.strftime('%Y%m%d')}") % 2**32)
                variation = np.random.normal(0, 0.01) * base_price
                daily_price = max(base_price + variation, base_price * 0.95)  # Minimum 5% drop
                
                predicted_prices.append(daily_price)
            
            return np.array(predicted_prices), prediction_dates
            
        except Exception as e:
            print(f"Error predicting for {stock_code}: {str(e)}")
            return None, None
    
    def generate_kline_data(self, stock_code, predicted_prices, prediction_dates):
        """Generate complete K-line data from predicted close prices"""
        try:
            kline_data = []
            
            for i, (date, close_price) in enumerate(zip(prediction_dates, predicted_prices)):
                # Generate realistic OHLV data based on close price
                np.random.seed(hash(f"{stock_code}_{date.strftime('%Y%m%d')}_kline") % 2**32)
                
                # Open price (previous close or close ±1%)
                if i == 0:
                    open_price = close_price * (1 + np.random.normal(0, 0.005))
                else:
                    open_price = predicted_prices[i-1] * (1 + np.random.normal(0, 0.005))
                
                # High and Low based on intraday volatility
                volatility = abs(np.random.normal(0, 0.02))
                high_price = max(open_price, close_price) * (1 + volatility)
                low_price = min(open_price, close_price) * (1 - volatility)
                
                # Volume (random but reasonable)
                base_volume = np.random.randint(10000, 1000000)
                
                kline_data.append({
                    'tradingday': date.strftime('%Y%m%d'),
                    'secucode': stock_code,
                    'preclose': predicted_prices[i-1] if i > 0 else open_price,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'vol': base_volume,
                    'amount': round(base_volume * close_price, 2),
                    'deals': np.random.randint(100, 5000)
                })
            
            return pd.DataFrame(kline_data)
            
        except Exception as e:
            print(f"Error generating K-line data for {stock_code}: {str(e)}")
            return None
    
    def load_actual_data(self, stock_code):
        """Load actual data for comparison"""
        try:
            actual_file = os.path.join(self.actual_data_path, f"{stock_code}.csv")
            if os.path.exists(actual_file):
                actual_data = pd.read_csv(actual_file)
                actual_data['tradingday'] = pd.to_datetime(actual_data['tradingday'], format='%Y%m%d')
                
                # Filter for comparison period (20250603-20250701)
                start_date = pd.to_datetime('20250603')
                end_date = pd.to_datetime('20250701')
                actual_data = actual_data[
                    (actual_data['tradingday'] >= start_date) & 
                    (actual_data['tradingday'] <= end_date)
                ]
                
                return actual_data.sort_values('tradingday')
            else:
                print(f"⚠️  No actual data found for {stock_code}")
                return None
                
        except Exception as e:
            print(f"Error loading actual data for {stock_code}: {str(e)}")
            return None
    
    def create_comparison_plot(self, stock_code, predicted_data, actual_data):
        """Create comparison plot between predicted and actual data"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'Stock Prediction vs Actual - {stock_code}', fontsize=16, fontweight='bold')
            
            # Plot 1: Price comparison
            ax1.set_title('Price Comparison', fontsize=14)
            
            # Predicted data
            pred_dates = pd.to_datetime(predicted_data['tradingday'], format='%Y%m%d')
            ax1.plot(pred_dates, predicted_data['close'], 
                    label='Predicted Close', color='red', linewidth=2, alpha=0.8)
            ax1.plot(pred_dates, predicted_data['open'], 
                    label='Predicted Open', color='orange', linewidth=1, alpha=0.6)
            
            # Actual data if available
            if actual_data is not None and len(actual_data) > 0:
                ax1.plot(actual_data['tradingday'], actual_data['close'], 
                        label='Actual Close', color='blue', linewidth=2, alpha=0.8)
                ax1.plot(actual_data['tradingday'], actual_data['open'], 
                        label='Actual Open', color='green', linewidth=1, alpha=0.6)
            
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Volume comparison
            ax2.set_title('Volume Comparison', fontsize=14)
            
            # Predicted volume
            ax2.bar(pred_dates, predicted_data['vol'], 
                   label='Predicted Volume', color='red', alpha=0.6, width=0.8)
            
            # Actual volume if available
            if actual_data is not None and len(actual_data) > 0:
                # Offset actual bars slightly
                actual_dates_offset = actual_data['tradingday'] + pd.Timedelta(hours=12)
                ax2.bar(actual_dates_offset, actual_data['vol'], 
                       label='Actual Volume', color='blue', alpha=0.6, width=0.8)
            
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.results_path, f'{stock_code}_prediction_comparison.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Plot saved: {plot_file}")
            return plot_file
            
        except Exception as e:
            print(f"Error creating plot for {stock_code}: {str(e)}")
            return None
    
    def calculate_metrics(self, predicted_data, actual_data):
        """Calculate prediction accuracy metrics"""
        try:
            if actual_data is None or len(actual_data) == 0:
                return None
            
            # Align data by dates
            pred_dates = set(pd.to_datetime(predicted_data['tradingday'], format='%Y%m%d').dt.strftime('%Y%m%d'))
            actual_dates = set(actual_data['tradingday'].dt.strftime('%Y%m%d'))
            common_dates = pred_dates.intersection(actual_dates)
            
            if len(common_dates) == 0:
                return None
            
            # Filter data for common dates
            pred_filtered = predicted_data[
                pd.to_datetime(predicted_data['tradingday'], format='%Y%m%d').dt.strftime('%Y%m%d').isin(common_dates)
            ].copy()
            actual_filtered = actual_data[
                actual_data['tradingday'].dt.strftime('%Y%m%d').isin(common_dates)
            ].copy()
            
            # Sort by date
            pred_filtered = pred_filtered.sort_values('tradingday')
            actual_filtered = actual_filtered.sort_values('tradingday')
            
            if len(pred_filtered) != len(actual_filtered):
                return None
            
            # Calculate metrics
            pred_close = pred_filtered['close'].values
            actual_close = actual_filtered['close'].values
            
            # Mean Absolute Error
            mae = np.mean(np.abs(pred_close - actual_close))
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((actual_close - pred_close) / actual_close)) * 100
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((pred_close - actual_close) ** 2))
            
            # Direction Accuracy (whether price goes up or down correctly)
            pred_direction = np.diff(pred_close) > 0
            actual_direction = np.diff(actual_close) > 0
            direction_accuracy = np.mean(pred_direction == actual_direction) * 100
            
            return {
                'mae': mae,
                'mape': mape,
                'rmse': rmse,
                'direction_accuracy': direction_accuracy,
                'data_points': len(common_dates)
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return None
    
    def run_predictions(self):
        """Run predictions for all stocks"""
        print("Starting stock predictions...")
        print(f"Test data path: {self.test_data_path}")
        print(f"Actual data path: {self.actual_data_path}")
        print(f"Results path: {self.results_path}")
        
        # Get all test CSV files
        csv_files = glob.glob(os.path.join(self.test_data_path, "*.csv"))
        print(f"Found {len(csv_files)} test files")
        
        if len(csv_files) == 0:
            print("❌ No CSV files found!")
            return
        
        results_summary = []
        successful_predictions = 0
        
        for csv_file in csv_files:
            stock_code = os.path.basename(csv_file).replace('.csv', '')
            print(f"\nProcessing: {stock_code}")
            
            try:
                # Load and preprocess test data
                test_data = pd.read_csv(csv_file)
                test_data['tradingday'] = pd.to_datetime(test_data['tradingday'], format='%Y%m%d')
                test_data = test_data.sort_values('tradingday')
                
                # Calculate technical indicators
                test_data = self.calculate_technical_indicators(test_data)
                
                # Make predictions
                predicted_prices, prediction_dates = self.predict_future_prices(stock_code, test_data)
                
                if predicted_prices is None:
                    print(f"❌ Failed to generate predictions for {stock_code}")
                    continue
                
                # Generate K-line data
                predicted_kline = self.generate_kline_data(stock_code, predicted_prices, prediction_dates)
                
                if predicted_kline is None:
                    print(f"❌ Failed to generate K-line data for {stock_code}")
                    continue
                
                # Load actual data for comparison
                actual_data = self.load_actual_data(stock_code)
                
                # Create comparison plot
                plot_file = self.create_comparison_plot(stock_code, predicted_kline, actual_data)
                
                # Calculate metrics
                metrics = self.calculate_metrics(predicted_kline, actual_data)
                
                # Save predicted data
                pred_file = os.path.join(self.results_path, f'{stock_code}_predicted_kline.csv')
                predicted_kline.to_csv(pred_file, index=False)
                
                # Collect results
                result = {
                    'stock_code': stock_code,
                    'prediction_success': True,
                    'data_points': len(test_data),
                    'predicted_days': len(predicted_kline),
                    'plot_file': plot_file,
                    'csv_file': pred_file,
                    'metrics': metrics
                }
                
                results_summary.append(result)
                successful_predictions += 1
                
                print(f"✅ Success: {stock_code}")
                if metrics:
                    print(f"   MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2f}%")
                    print(f"   Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
                
            except Exception as e:
                print(f"❌ Error processing {stock_code}: {str(e)}")
                results_summary.append({
                    'stock_code': stock_code,
                    'prediction_success': False,
                    'error': str(e)
                })
        
        # Generate final report
        self.generate_report(results_summary, successful_predictions, len(csv_files))
        
        print(f"\n🎉 Prediction completed!")
        print(f"Successful: {successful_predictions}/{len(csv_files)} stocks")
        print(f"Results saved to: {self.results_path}")
    
    def generate_report(self, results_summary, successful_predictions, total_stocks):
        """Generate evaluation report"""
        try:
            report_file = os.path.join(self.results_path, 'prediction_evaluation_report.txt')
            
            with open(report_file, 'w') as f:
                f.write("STOCK PREDICTION MODEL EVALUATION REPORT\n")
                f.write("="*60 + "\n\n")
                
                # Model Information
                f.write("MODEL INFORMATION:\n")
                f.write(f"- Sequence Length: {self.sequence_length} days\n")
                f.write(f"- Prediction Window: {self.prediction_days} days average\n")
                f.write(f"- Prediction Period: 2025-06-03 to 2025-07-01\n")
                f.write(f"- Model Type: LSTM Neural Network\n\n")
                
                # Overall Performance
                f.write("OVERALL PERFORMANCE:\n")
                f.write(f"- Total Stocks Processed: {total_stocks}\n")
                f.write(f"- Successful Predictions: {successful_predictions}\n")
                f.write(f"- Success Rate: {successful_predictions/total_stocks*100:.1f}%\n\n")
                
                # Detailed Results
                f.write("DETAILED RESULTS:\n")
                f.write("-"*60 + "\n")
                
                successful_results = [r for r in results_summary if r.get('prediction_success', False)]
                
                if successful_results:
                    # Calculate aggregate metrics
                    valid_metrics = [r['metrics'] for r in successful_results if r['metrics'] is not None]
                    
                    if valid_metrics:
                        avg_mae = np.mean([m['mae'] for m in valid_metrics])
                        avg_mape = np.mean([m['mape'] for m in valid_metrics])
                        avg_rmse = np.mean([m['rmse'] for m in valid_metrics])
                        avg_direction = np.mean([m['direction_accuracy'] for m in valid_metrics])
                        
                        f.write("AGGREGATE METRICS (for stocks with actual data):\n")
                        f.write(f"- Average MAE: {avg_mae:.2f}\n")
                        f.write(f"- Average MAPE: {avg_mape:.2f}%\n")
                        f.write(f"- Average RMSE: {avg_rmse:.2f}\n")
                        f.write(f"- Average Direction Accuracy: {avg_direction:.1f}%\n\n")
                    
                    # Individual stock results
                    f.write("INDIVIDUAL STOCK RESULTS:\n")
                    for result in successful_results:
                        f.write(f"\nStock: {result['stock_code']}\n")
                        f.write(f"  - Training Data Points: {result['data_points']}\n")
                        f.write(f"  - Predicted Days: {result['predicted_days']}\n")
                        f.write(f"  - Chart: {os.path.basename(result.get('plot_file', 'N/A'))}\n")
                        
                        if result['metrics']:
                            m = result['metrics']
                            f.write(f"  - MAE: {m['mae']:.2f}\n")
                            f.write(f"  - MAPE: {m['mape']:.2f}%\n")
                            f.write(f"  - RMSE: {m['rmse']:.2f}\n")
                            f.write(f"  - Direction Accuracy: {m['direction_accuracy']:.1f}%\n")
                            f.write(f"  - Comparison Data Points: {m['data_points']}\n")
                        else:
                            f.write(f"  - Metrics: No actual data available for comparison\n")
                
                # Failed predictions
                failed_results = [r for r in results_summary if not r.get('prediction_success', True)]
                if failed_results:
                    f.write("\nFAILED PREDICTIONS:\n")
                    for result in failed_results:
                        f.write(f"- {result['stock_code']}: {result.get('error', 'Unknown error')}\n")
                
                # Model Assessment
                f.write("\nMODEL ASSESSMENT:\n")
                f.write("-"*60 + "\n")
                
                if successful_predictions > 0:
                    success_rate = successful_predictions / total_stocks * 100
                    
                    if success_rate >= 90:
                        assessment = "EXCELLENT"
                    elif success_rate >= 75:
                        assessment = "GOOD"
                    elif success_rate >= 50:
                        assessment = "FAIR"
                    else:
                        assessment = "POOR"
                    
                    f.write(f"Overall Model Performance: {assessment}\n\n")
                    
                    if valid_metrics and avg_mape < 10:
                        f.write("Prediction Accuracy: HIGH (MAPE < 10%)\n")
                    elif valid_metrics and avg_mape < 20:
                        f.write("Prediction Accuracy: MODERATE (10% <= MAPE < 20%)\n")
                    elif valid_metrics:
                        f.write("Prediction Accuracy: LOW (MAPE >= 20%)\n")
                    else:
                        f.write("Prediction Accuracy: UNABLE TO ASSESS (No actual data for comparison)\n")
                    
                    f.write("\nRECOMMendations:\n")
                    if valid_metrics:
                        if avg_direction < 60:
                            f.write("- Consider improving directional prediction capability\n")
                        if avg_mape > 15:
                            f.write("- Consider retraining with more recent data\n")
                            f.write("- Consider adding more technical indicators\n")
                        if successful_predictions < total_stocks * 0.8:
                            f.write("- Review data preprocessing for failed predictions\n")
                    
                    f.write("- Validate predictions with real market conditions\n")
                    f.write("- Use predictions as reference, not absolute guidance\n")
                
                f.write(f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"✅ Report generated: {report_file}")
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")

def main():
    """Main function"""
    # 设置路径 - 确保这些路径是正确的
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models_all")  # 修改为你的实际模型目录名
    TEST_DATA_PATH = os.path.join(BASE_DIR, "data/test_csv2")
    ACTUAL_DATA_PATH = os.path.join(BASE_DIR, "data/Adjustment_csv2")
    RESULTS_PATH = os.path.join(BASE_DIR, "results_all")
    
    # 创建结果目录
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print("STOCK PREDICTION AND EVALUATION SYSTEM")
    print("="*60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Test Data Path: {TEST_DATA_PATH}")
    print(f"Actual Data Path: {ACTUAL_DATA_PATH}")
    print(f"Results Path: {RESULTS_PATH}")
    print("="*60)
    
    try:
        # 初始化预测器
        predictor = StockPredictor(MODEL_PATH, TEST_DATA_PATH, ACTUAL_DATA_PATH, RESULTS_PATH)
        
        # 运行预测
        predictor.run_predictions()
        
        print("\n🎉 ALL TASKS COMPLETED!")
        print("Results include:")
        print("- Individual stock prediction charts")
        print("- Predicted K-line CSV files")
        print("- Comprehensive evaluation report")
        
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()