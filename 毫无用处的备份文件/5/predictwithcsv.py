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

class StockPredictor:
    def __init__(self):
        # Path settings (using relative paths)
        self.model_path = "./models2"
        self.test_data_path = "./data/test_csv2"
        self.real_data_path = "./data/Adjustment_csv2"
        self.results_path = "./results"
        
        # Create results directory
        os.makedirs(self.results_path, exist_ok=True)
        
        # Model parameters
        self.sequence_length = 60
        self.features = ['open', 'high', 'low', 'close', 'vol', 'amount']
        self.target = 'close'
        
        # Prediction date range
        self.pred_start_date = '2025-06-03'
        self.pred_end_date = '2025-07-01'

    def get_trading_days(self, start_date, end_date):
        """Generate trading days list (excluding weekends)"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        trading_days = []
        current_date = start
        while current_date <= end:
            # Exclude weekends (0=Monday, 6=Sunday)
            if current_date.weekday() < 5:
                trading_days.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
        return trading_days

    def build_model_struct(self, input_shape):
        """Rebuild model structure (consistent with training)"""
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
        """Load model and scalers"""
        try:
            model_file = f"{self.model_path}/{stock_code}_model.h5"
            best_model_file = f"{self.model_path}/{stock_code}_best.h5"
            scaler_file = f"{self.model_path}/{stock_code}_scalers.pkl"
            
            if not os.path.exists(scaler_file):
                return None, None
            
            # First try to load model directly
            model = None
            for model_path in [best_model_file, model_file]:
                if os.path.exists(model_path):
                    try:
                        # Method 1: Use custom_objects to resolve function recognition
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
                            # Method 2: Rebuild model structure and load weights
                            print(f"Attempting to rebuild model structure for: {stock_code}")
                            model = self.build_model_struct((self.sequence_length, len(self.features)))
                            model.load_weights(model_path)
                            break
                        except Exception as load_error2:
                            print(f"Failed to load {model_path}: {str(load_error2)}")
                            continue
            
            if model is None:
                return None, None
                
            with open(scaler_file, 'rb') as f:
                feature_scaler, target_scaler = pickle.load(f)
                
            return model, (feature_scaler, target_scaler)
            
        except Exception as e:
            print(f"Error loading model {stock_code}: {str(e)}")
            return None, None

    def load_test_data(self, stock_code):
        """Load test data"""
        try:
            file_path = os.path.join(self.test_data_path, f"{stock_code}.csv")
            if not os.path.exists(file_path):
                return None
            
            df = pd.read_csv(file_path)
            df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y%m%d')
            df = df.sort_values('tradingday').reset_index(drop=True)
            
            return df
        except Exception as e:
            print(f"Error loading test data {stock_code}: {str(e)}")
            return None

    def load_real_data(self, stock_code):
        """Load real data"""
        try:
            file_path = os.path.join(self.real_data_path, f"{stock_code}.csv")
            if not os.path.exists(file_path):
                return None
            
            df = pd.read_csv(file_path)
            df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y%m%d')
            df = df.sort_values('tradingday').reset_index(drop=True)
            
            # Filter data for prediction period
            start_date = datetime.strptime(self.pred_start_date, '%Y-%m-%d')
            end_date = datetime.strptime(self.pred_end_date, '%Y-%m-%d')
            df_filtered = df[(df['tradingday'] >= start_date) & (df['tradingday'] <= end_date)]
            
            return df_filtered
        except Exception as e:
            print(f"Error loading real data {stock_code}: {str(e)}")
            return None

    def predict_future_prices(self, model, test_data, scalers, prediction_days):
        """Predict future stock prices"""
        feature_scaler, target_scaler = scalers
        
        # Get last sequence_length days as base input
        recent_data = test_data[self.features].tail(self.sequence_length).values
        recent_scaled = feature_scaler.transform(recent_data)
        
        predictions = []
        current_sequence = recent_scaled.copy()
        
        for _ in range(prediction_days):
            # Prepare prediction input
            X_pred = current_sequence.reshape(1, self.sequence_length, len(self.features))
            
            # Predict next day's closing price
            pred_scaled = model.predict(X_pred, verbose=0)
            pred_price = target_scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred_price)
            
            # Update sequence (using predicted close price for next day's data)
            last_row = current_sequence[-1].copy()
            
            # Simple assumption: open=previous close, other prices fluctuate around close
            pred_scaled_price = pred_scaled[0][0]
            
            # Update close value in last row
            last_row[3] = pred_scaled_price  # position of 'close' in features
            
            # Sliding window: remove first day, add new predicted day
            current_sequence = np.vstack([current_sequence[1:], last_row])
            
        return predictions

    def generate_prediction_data(self, stock_code, predictions, trading_days, last_real_data):
        """Generate complete prediction data"""
        pred_data = []
        
        for i, (day, close_price) in enumerate(zip(trading_days, predictions)):
            # Simple price generation strategy
            if i == 0 and last_real_data is not None:
                # First day uses last real data as reference
                open_price = last_real_data['close']
            else:
                # Subsequent days' open equals previous close
                open_price = predictions[i-1] if i > 0 else close_price
            
            # Generate high/low prices (assuming ±2% fluctuation)
            volatility = 0.02
            high_price = close_price * (1 + np.random.uniform(0, volatility))
            low_price = close_price * (1 - np.random.uniform(0, volatility))
            
            # Ensure price logic is correct
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            pred_data.append({
                'tradingday': day,
                'secucode': stock_code,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'vol': 0,  # Volume prediction is complex, set to 0
                'amount': 0  # Turnover prediction is complex, set to 0
            })
        
        return pd.DataFrame(pred_data)

    def save_prediction_csv(self, stock_code, pred_data):
        """Save prediction results to CSV file"""
        csv_path = f"{self.results_path}/{stock_code}_predictions.csv"
        pred_data.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Saved predictions for {stock_code} to: {csv_path}")

    def plot_comparison(self, stock_code, pred_data, real_data, metrics):
        """Plot comparison between predicted and real data"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{stock_code} Stock Prediction vs Real Data Comparison', fontsize=16, fontweight='bold')
        
        # Convert date format
        pred_data['date'] = pd.to_datetime(pred_data['tradingday'], format='%Y%m%d')
        real_data['date'] = real_data['tradingday']
        
        # 1. Close price comparison
        ax1.plot(pred_data['date'], pred_data['close'], 'b-', label='Predicted Price', linewidth=2)
        ax1.plot(real_data['date'], real_data['close'], 'r-', label='Real Price', linewidth=2)
        ax1.set_title('Close Price Comparison')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Open price comparison
        ax2.plot(pred_data['date'], pred_data['open'], 'b-', label='Predicted Open', linewidth=2)
        ax2.plot(real_data['date'], real_data['open'], 'r-', label='Real Open', linewidth=2)
        ax2.set_title('Open Price Comparison')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. High/Low price comparison
        ax3.plot(pred_data['date'], pred_data['high'], 'b-', label='Predicted High', alpha=0.7)
        ax3.plot(pred_data['date'], pred_data['low'], 'b--', label='Predicted Low', alpha=0.7)
        ax3.plot(real_data['date'], real_data['high'], 'r-', label='Real High', alpha=0.7)
        ax3.plot(real_data['date'], real_data['low'], 'r--', label='Real Low', alpha=0.7)
        ax3.fill_between(pred_data['date'], pred_data['low'], pred_data['high'], 
                        alpha=0.2, color='blue', label='Predicted Range')
        ax3.fill_between(real_data['date'], real_data['low'], real_data['high'], 
                        alpha=0.2, color='red', label='Real Range')
        ax3.set_title('High and Low Price Comparison')
        ax3.set_ylabel('Price')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Prediction error analysis
        if len(pred_data) == len(real_data):
            errors = np.abs(pred_data['close'].values - real_data['close'].values)
            ax4.bar(range(len(errors)), errors, alpha=0.7, color='orange')
            ax4.set_title('Daily Prediction Absolute Error')
            ax4.set_xlabel('Trading Day')
            ax4.set_ylabel('Absolute Error')
            ax4.grid(True, alpha=0.3)
            
            # Add mean error line
            mean_error = np.mean(errors)
            ax4.axhline(y=mean_error, color='red', linestyle='--', 
                       label=f'Mean Error: {mean_error:.2f}')
            ax4.legend()
        
        # Add evaluation metrics text
        textstr = f'''Evaluation Metrics:
MSE: {metrics['mse']:.6f}
RMSE: {metrics['rmse']:.4f}
MAE: {metrics['mae']:.4f}
MAPE: {metrics['mape']:.2f}%
R²: {metrics['r2']:.4f}'''
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        fig.text(0.02, 0.02, textstr, fontsize=10, bbox=props)
        
        plt.tight_layout()
        
        # Save image
        plt.savefig(f"{self.results_path}/{stock_code}_prediction_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot for {stock_code}")

    def calculate_metrics(self, pred_values, real_values):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(real_values, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(real_values, pred_values)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((real_values - pred_values) / real_values)) * 100
        
        # R² coefficient of determination
        r2 = r2_score(real_values, pred_values)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }

    def predict_single_stock(self, stock_code):
        """Predict single stock"""
        print(f"Predicting stock {stock_code}...")
        
        # Load model and scalers
        model, scalers = self.load_model_and_scalers(stock_code)
        if model is None:
            print(f"Failed to load model for {stock_code}")
            return None
        
        # Load test data
        test_data = self.load_test_data(stock_code)
        if test_data is None:
            print(f"Failed to load test data for {stock_code}")
            return None
        
        # Load real data
        real_data = self.load_real_data(stock_code)
        if real_data is None:
            print(f"Failed to load real data for {stock_code}")
            return None
        
        # Generate trading days list
        trading_days = self.get_trading_days(self.pred_start_date, self.pred_end_date)
        
        # Make predictions
        predictions = self.predict_future_prices(model, test_data, scalers, len(trading_days))
        
        # Generate prediction DataFrame
        last_real_data = test_data.iloc[-1] if not test_data.empty else None
        pred_data = self.generate_prediction_data(stock_code, predictions, trading_days, last_real_data)
        
        # Save prediction results to CSV
        self.save_prediction_csv(stock_code, pred_data)
        
        # Ensure real data and prediction dates match
        real_data_filtered = real_data[real_data['tradingday'].dt.strftime('%Y%m%d').isin(trading_days)]
        
        if len(real_data_filtered) == 0:
            print(f"No real data available for {stock_code} in prediction period")
            return None
        
        # Calculate evaluation metrics
        if len(pred_data) == len(real_data_filtered):
            metrics = self.calculate_metrics(
                pred_data['close'].values,
                real_data_filtered['close'].values
            )
        else:
            print(f"Warning: {stock_code} prediction and real data length mismatch")
            # Use shorter length for calculation
            min_len = min(len(pred_data), len(real_data_filtered))
            metrics = self.calculate_metrics(
                pred_data['close'].head(min_len).values,
                real_data_filtered['close'].head(min_len).values
            )
        
        # Plot comparison
        self.plot_comparison(stock_code, pred_data, real_data_filtered, metrics)
        
        return {
            'stock_code': stock_code,
            'metrics': metrics,
            'pred_data': pred_data,
            'real_data': real_data_filtered,
            'prediction_days': len(trading_days)
        }

    def run_all_predictions(self):
        """Run predictions for all stocks"""
        # Get all trained models
        model_files = glob.glob(os.path.join(self.model_path, "*_model.h5"))
        stock_codes = [os.path.basename(f).replace('_model.h5', '') for f in model_files]
        
        if not stock_codes:
            print("No trained models found")
            return
        
        print(f"Found {len(stock_codes)} trained models")
        
        results = []
        successful_predictions = 0
        failed_predictions = 0
        
        for i, stock_code in enumerate(stock_codes, 1):
            print(f"\nProgress: {i}/{len(stock_codes)}")
            result = self.predict_single_stock(stock_code)
            
            if result:
                results.append(result)
                successful_predictions += 1
            else:
                failed_predictions += 1
        
        # Generate summary report
        self.generate_summary_report(results, successful_predictions, failed_predictions)
        
        print(f"\nPrediction completed!")
        print(f"Successful predictions: {successful_predictions} stocks")
        print(f"Failed predictions: {failed_predictions} stocks")
        print(f"Results saved to: {self.results_path}")

    def generate_summary_report(self, results, success, failed):
        """Generate prediction report"""
        report_path = os.path.join(self.results_path, "prediction_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Stock Price Prediction Model Evaluation Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Prediction period: {self.pred_start_date} to {self.pred_end_date}\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total stocks: {success + failed}\n")
            f.write(f"Successful predictions: {success}\n")
            f.write(f"Failed predictions: {failed}\n")
            f.write(f"Success rate: {success/(success + failed)*100:.1f}%\n\n")
            
            if results:
                # Calculate overall metrics
                all_mse = [r['metrics']['mse'] for r in results]
                all_rmse = [r['metrics']['rmse'] for r in results]
                all_mae = [r['metrics']['mae'] for r in results]
                all_mape = [r['metrics']['mape'] for r in results]
                all_r2 = [r['metrics']['r2'] for r in results]
                
                f.write("Overall Performance Metrics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Mean MSE: {np.mean(all_mse):.6f} (±{np.std(all_mse):.6f})\n")
                f.write(f"Mean RMSE: {np.mean(all_rmse):.4f} (±{np.std(all_rmse):.4f})\n")
                f.write(f"Mean MAE: {np.mean(all_mae):.4f} (±{np.std(all_mae):.4f})\n")
                f.write(f"Mean MAPE: {np.mean(all_mape):.2f}% (±{np.std(all_mape):.2f}%)\n")
                f.write(f"Mean R²: {np.mean(all_r2):.4f} (±{np.std(all_r2):.4f})\n\n")
                
                # Performance classification
                excellent = sum(1 for r in results if r['metrics']['mape'] < 5)
                good = sum(1 for r in results if 5 <= r['metrics']['mape'] < 10)
                fair = sum(1 for r in results if 10 <= r['metrics']['mape'] < 20)
                poor = sum(1 for r in results if r['metrics']['mape'] >= 20)
                
                f.write("Prediction Accuracy Classification (based on MAPE):\n")
                f.write("-" * 30 + "\n")
                f.write(f"Excellent (<5%): {excellent} stocks ({excellent/len(results)*100:.1f}%)\n")
                f.write(f"Good (5-10%): {good} stocks ({good/len(results)*100:.1f}%)\n")
                f.write(f"Fair (10-20%): {fair} stocks ({fair/len(results)*100:.1f}%)\n")
                f.write(f"Poor (>20%): {poor} stocks ({poor/len(results)*100:.1f}%)\n\n")
                
                # Best and worst performance
                best_stock = min(results, key=lambda x: x['metrics']['mape'])
                worst_stock = max(results, key=lambda x: x['metrics']['mape'])
                
                f.write("Best Prediction Performance:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Stock Code: {best_stock['stock_code']}\n")
                f.write(f"MAPE: {best_stock['metrics']['mape']:.2f}%\n")
                f.write(f"RMSE: {best_stock['metrics']['rmse']:.4f}\n")
                f.write(f"R²: {best_stock['metrics']['r2']:.4f}\n\n")
                
                f.write("Worst Prediction Performance:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Stock Code: {worst_stock['stock_code']}\n")
                f.write(f"MAPE: {worst_stock['metrics']['mape']:.2f}%\n")
                f.write(f"RMSE: {worst_stock['metrics']['rmse']:.4f}\n")
                f.write(f"R²: {worst_stock['metrics']['r2']:.4f}\n\n")
                
                # Detailed results
                f.write("Detailed Prediction Results:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Stock':<10} {'MSE':<12} {'RMSE':<8} {'MAE':<8} {'MAPE(%)':<10} {'R²':<8}\n")
                f.write("-" * 80 + "\n")
                
                for result in sorted(results, key=lambda x: x['metrics']['mape']):
                    metrics = result['metrics']
                    f.write(f"{result['stock_code']:<10} {metrics['mse']:<12.6f} "
                           f"{metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} "
                           f"{metrics['mape']:<10.2f} {metrics['r2']:<8.4f}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("Report Notes:\n")
                f.write("MSE: Mean Squared Error, lower is better\n")
                f.write("RMSE: Root Mean Squared Error, lower is better\n")
                f.write("MAE: Mean Absolute Error, lower is better\n")
                f.write("MAPE: Mean Absolute Percentage Error, lower is better\n")
                f.write("R²: Coefficient of Determination, closer to 1 is better\n")
        
        print(f"Prediction report saved to: {report_path}")

def main():
    """Main function"""    
    predictor = StockPredictor()
    print("Starting stock prediction and evaluation...")
    print(f"Prediction period: {predictor.pred_start_date} to {predictor.pred_end_date}")
    predictor.run_all_predictions()

if __name__ == "__main__":
    main()