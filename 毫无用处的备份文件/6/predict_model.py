import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
warnings.filterwarnings('ignore')

# ============================== 路径配置 ==============================
class PredictionConfig:
    def __init__(self):
        # 路径配置 - 根据实际文件结构设置
        self.test_data_path = "data/test_csv3"         # 测试数据路径
        self.model_path = "models2"                    # 训练好的模型路径
        self.results_path = "results3"                 # 结果保存路径
        self.second_path = "data/Adjustment_csv3/second"  # second路径（昨天信息）
        
        # 预测配置
        self.sequence_length = 100  # 与训练时保持一致
        self.prediction_days = 20   # 预测未来天数
        self.target_column = 'close'
        
        # 滚动预测配置
        self.rolling_window = 1     # 每次滚动的步长（1表示每次预测1天然后滚动）
        self.min_history_days = 150 # 最少需要的历史数据天数
        
        # 评估配置
        self.confidence_interval = 0.95  # 置信区间

config = PredictionConfig()

# 板块股票字典（与训练时保持一致）
板块股票 = {
    "银行": ["000001","002142","600000","600015","600016","600036","600919","600926","601009","601166","601169","601229","601288","601328","601398","601658","601818","601838","601939","601988","601998"],
    "证券": ["000166","000776","002736","600030","600837","600918","600958","600999","601066","601211","601236","601377","601688","601788","601878","601881","601901","601995"],
    "保险": ["601318","601319","601336","601601","601628"],
    "白酒及酒类": ["000568","000596","000858","002304","600132","600519","600600","600809","603369"],
    "食品饮料": ["000895","300999","600887","603288","603899","605499"],
    "家电": ["000333","000651","600690"],
    "房地产": ["000002","000069","001979","600048","600383","600606","601155"],
    "汽车": ["000338","000625","000800","002594","002920","600104","600741","601238","601633","601689","601799"],
    "新能源": ["002460","002466","002756","300014","300207","300274","300750","300763","300769","300919","601012","601615","601865","603185","603806","605117","688223","688303","688599"],
    "医药生物": ["000538","000661","000963","002007","002252","002821","300015","300122","300142","300347","300601","300759","300760","300896","300957","600085","600196","600276","600332","600436","600763","601607","603259","603392","688065","688363"],
    "电子科技": ["000063","000100","000725","000733","000938","000977","002049","002129","002179","002230","002236","002241","002371","002414","002415","002475","002841","002916","002938","300223","300316","300408","300433","300450","300454","300496","300628","300661","300751","300782","600183","600460","600570","600584","600745","600845","601138","603019","603290","603501","603986","688008","688012","688036","688111","688126","688187","688396","688561","688981"],
    "通信": ["600050","600588","600941","601360","601698","601728"],
    "钢铁有色": ["000408","000708","000792","002601","600010","600019","600111","600176","600219","600362","600547","601600","601899","603799","603993","688005"],
    "能源电力": ["000723","000983","001289","003816","600011","600025","600028","600089","600188","600406","600438","600674","600732","600795","600803","600875","600884","600886","600900","600905","600989","601088","601699","601808","601857","601868","601877","601898","601985"],
    "化工": ["000301","002001","002064","002493","002648","002709","002812","600309","600346","600426","603260","603659"],
    "建材建筑": ["000786","000877","002271","600039","600585","601117","601186","601390","601618","601668","601669","601800"],
    "机械设备": ["000157","000425","300124","600031","601100"],
    "交通运输": ["002352","002120","600009","600018","600029","600115","600233","600754","601006","601021","601111","601766","601816","601872","601919"],
    "军工航天": ["000768","600150","600760","600893","601989"],
    "农林牧渔": ["000876","002311","002714","300498"],
    "传媒娱乐": ["002027","002555","300033","300059","300413"],
    "消费零售": ["002180","002410","300979","600660","601888","603195","603486","603833"],
    "其他": ["000617","002074","002202","002459","002050","600061","601216","601225"]
}

# ============================== 模型定义 ==============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, 
                dim_feedforward=512, dropout=0.1, output_dim=1):
        super(StockTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_dim)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        output = self.output_projection(x)
        return output

# ============================== 数据处理函数 ==============================
def load_and_preprocess_data(file_path, stock_code):
    """加载和预处理单只股票数据"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='gbk')
    
    # 自动识别日期列
    if 'Date' in df.columns:
        date_col = 'Date'
    elif '日期' in df.columns:
        date_col = '日期'
    else:
        raise ValueError(f"{file_path} 未找到日期列")
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # 定义特征列
    price_features = ['open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'change_ratio']
    news_features = [
        'Newsnum_Title_news1', 'Newsnum_Cont_news1', 'Posnews_All_news1',
        'Neunews_All_news1', 'Negnews_All_news1', 'Posnews_Ori_news1',
        'Neunews_Ori_news1', 'Negnews_Ori_news1', 'Newsnum_Title_news2',
        'Newsnum_Cont_news2', 'Posnews_All_news2', 'Neunews_All_news2',
        'Negnews_All_news2', 'Posnews_Ori_news2', 'Neunews_Ori_news2',
        'Negnews_Ori_news2'
    ]
    
    # 处理缺失值
    for col in price_features:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    for col in news_features:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(0)
    
    # 创建技术指标
    df = create_technical_indicators(df)
    
    return df

def create_technical_indicators(df):
    """创建技术指标"""
    # 移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # 布林带
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # 填充技术指标的NaN值
    technical_cols = ['MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
                    'BB_middle', 'BB_upper', 'BB_lower']
    for col in technical_cols:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    return df

def get_stock_sector(stock_code):
    """根据股票代码获取所属板块"""
    for sector, codes in 板块股票.items():
        if stock_code in codes:
            return sector
    return "其他"

def load_model_and_scalers(sector_name):
    """加载指定板块的模型和标准化器"""
    model_dir = os.path.join(config.model_path, sector_name)
    
    # 加载模型配置和状态
    model_file = os.path.join(model_dir, f'{sector_name}_model.pth')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    
    checkpoint = torch.load(model_file, map_location='cpu')
    model_config = checkpoint['model_config']
    
    # 创建模型
    model = StockTransformer(
        input_dim=model_config['input_dim'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        dropout=model_config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载标准化器  
    scaler_X = joblib.load(os.path.join(model_dir, f'{sector_name}_scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(model_dir, f'{sector_name}_scaler_y.pkl'))
    
    return model, scaler_X, scaler_y, checkpoint

def create_prediction_sequences(df, sequence_length, end_idx=None):
    """创建用于预测的序列"""
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'change_ratio',
        'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal',
        'BB_middle', 'BB_upper', 'BB_lower',
        'Newsnum_Title_news1', 'Newsnum_Cont_news1', 'Posnews_All_news1',
        'Neunews_All_news1', 'Negnews_All_news1', 'Posnews_Ori_news1',
        'Neunews_Ori_news1', 'Negnews_Ori_news1', 'Newsnum_Title_news2',
        'Newsnum_Cont_news2', 'Posnews_All_news2', 'Neunews_All_news2',
        'Negnews_All_news2', 'Posnews_Ori_news2', 'Neunews_Ori_news2',
        'Negnews_Ori_news2'
    ]
    
    # 过滤存在的列
    available_features = [col for col in feature_columns if col in df.columns]
    
    if end_idx is None:
        end_idx = len(df)
    
    if end_idx < sequence_length:
        raise ValueError(f"数据长度不足，需要至少 {sequence_length} 条记录")
    
    # 获取指定结束位置前sequence_length天的数据作为输入
    start_idx = end_idx - sequence_length
    sequences = df[available_features].iloc[start_idx:end_idx].values
    
    return sequences.reshape(1, sequence_length, -1), available_features

def predict_single_step(model, scaler_X, scaler_y, sequences):
    """单步预测"""
    try:
        # 标准化输入数据
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = scaler_X.transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(sequences.shape)
        
        # 预测
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        with torch.no_grad():
            sequences_tensor = torch.FloatTensor(sequences_scaled).to(device)
            prediction_scaled = model(sequences_tensor).cpu().numpy()
        
        # 反标准化预测结果
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
        
        return prediction, True, None
        
    except Exception as e:
        return None, False, str(e)

def rolling_prediction(stock_code, model, scaler_X, scaler_y, df):
    """滚动预测函数"""
    print(f"开始对股票 {stock_code} 进行滚动预测...")
    
    # 确保有足够的数据
    total_days = len(df)
    if total_days < config.min_history_days + config.prediction_days:
        raise ValueError(f"数据不足：需要至少 {config.min_history_days + config.prediction_days} 天，实际 {total_days} 天")
    
    # 计算预测的起始和结束位置
    prediction_start_idx = config.min_history_days
    prediction_end_idx = total_days - config.prediction_days + 1
    
    predictions = []
    actuals = []
    dates = []
    prediction_errors = []
    
    # 获取日期列名
    date_col = 'Date' if 'Date' in df.columns else '日期'
    
    # 滚动预测循环
    for current_idx in range(prediction_start_idx, prediction_end_idx, config.rolling_window):
        try:
            # 创建当前预测的输入序列（使用当前时间点之前的数据）
            sequences, feature_names = create_prediction_sequences(
                df, config.sequence_length, current_idx
            )
            
            # 进行预测
            prediction, success, error_msg = predict_single_step(
                model, scaler_X, scaler_y, sequences
            )
            
            if not success:
                print(f"预测失败 (索引 {current_idx}): {error_msg}")
                prediction_errors.append({
                    'index': current_idx,
                    'error': error_msg
                })
                continue
            
            # 获取实际值（预测未来第prediction_days天的价格）
            actual_idx = current_idx + config.prediction_days - 1
            if actual_idx >= len(df):
                break
                
            actual_price = df[config.target_column].iloc[actual_idx]
            actual_date = df[date_col].iloc[actual_idx]
            
            # 记录预测结果
            predictions.append(prediction)
            actuals.append(actual_price)
            dates.append(actual_date)
            
            if len(predictions) % 10 == 0:  # 每10步打印一次进度
                print(f"已完成 {len(predictions)} 步预测...")
                
        except Exception as e:
            print(f"预测步骤出错 (索引 {current_idx}): {str(e)}")
            prediction_errors.append({
                'index': current_idx,
                'error': str(e)
            })
            continue
    
    if not predictions:
        raise ValueError("没有成功的预测结果")
    
    print(f"股票 {stock_code} 滚动预测完成，共 {len(predictions)} 个预测点")
    
    return {
        'predictions': np.array(predictions),
        'actuals': np.array(actuals),
        'dates': dates,
        'errors': prediction_errors,
        'total_steps': len(predictions)
    }

def calculate_rolling_metrics(predictions, actuals):
    """计算滚动预测的评估指标"""
    if len(predictions) == 0 or len(actuals) == 0:
        return {}
    
    # 基本回归指标
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # 方向准确性 (预测涨跌方向的准确率)
    if len(actuals) > 1:
        actual_direction = np.diff(actuals) > 0
        pred_direction = np.diff(predictions) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100 if len(actual_direction) > 0 else 0
    else:
        direction_accuracy = 0
    
    # 累计收益率比较
    actual_returns = (actuals[-1] - actuals[0]) / actuals[0] * 100 if len(actuals) > 0 else 0
    pred_returns = (predictions[-1] - predictions[0]) / predictions[0] * 100 if len(predictions) > 0 else 0
    
    # 趋势一致性 (整体趋势方向是否一致)
    actual_trend = 1 if actuals[-1] > actuals[0] else -1 if actuals[-1] < actuals[0] else 0
    pred_trend = 1 if predictions[-1] > predictions[0] else -1 if predictions[-1] < predictions[0] else 0
    trend_consistency = actual_trend == pred_trend
    
    # 最大绝对误差
    max_error = np.max(np.abs(actuals - predictions))
    
    # 平均价格
    avg_actual_price = np.mean(actuals)
    avg_pred_price = np.mean(predictions)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy,
        'Actual_Returns': actual_returns,
        'Predicted_Returns': pred_returns,
        'Trend_Consistency': trend_consistency,
        'Max_Error': max_error,
        'Avg_Actual_Price': avg_actual_price,
        'Avg_Predicted_Price': avg_pred_price,
        'Total_Predictions': len(predictions)
    }

def save_rolling_results(stock_code, sector, results, metrics):
    """保存滚动预测结果"""
    # 创建详细结果DataFrame
    result_df = pd.DataFrame({
        'Date': results['dates'],
        'Actual_Price': results['actuals'],
        'Predicted_Price': results['predictions'],
        'Absolute_Error': np.abs(results['actuals'] - results['predictions']),
        'Percentage_Error': np.abs((results['actuals'] - results['predictions']) / results['actuals']) * 100,
        'Stock_Code': stock_code,
        'Sector': sector
    })
    
    # 添加移动平均误差（平滑误差趋势）
    result_df['MA5_Error'] = result_df['Absolute_Error'].rolling(window=5).mean()
    result_df['MA10_Error'] = result_df['Absolute_Error'].rolling(window=10).mean()
    
    # 保存详细结果
    result_file = os.path.join(config.results_path, f"{stock_code}_rolling_prediction.csv")
    result_df.to_csv(result_file, index=False, encoding='utf-8')
    
    # 创建指标摘要
    metrics_df = pd.DataFrame([metrics])
    metrics_df['Stock_Code'] = stock_code
    metrics_df['Sector'] = sector
    
    return result_df, metrics_df

def plot_rolling_prediction(stock_code, result_df, save_plot=True):
    """绘制滚动预测结果图表"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 上图：价格预测对比
        ax1.plot(range(len(result_df)), result_df['Actual_Price'], 
                label='实际价格', color='blue', linewidth=2)
        ax1.plot(range(len(result_df)), result_df['Predicted_Price'], 
                label='预测价格', color='red', linewidth=2, alpha=0.8)
        ax1.set_title(f'{stock_code} 滚动预测结果对比')
        ax1.set_xlabel('预测步数')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：预测误差
        ax2.plot(range(len(result_df)), result_df['Absolute_Error'], 
                label='绝对误差', color='orange', alpha=0.7)
        if 'MA5_Error' in result_df.columns:
            ax2.plot(range(len(result_df)), result_df['MA5_Error'], 
                    label='5日平均误差', color='green', linewidth=2)
        ax2.set_title(f'{stock_code} 预测误差变化')
        ax2.set_xlabel('预测步数')
        ax2.set_ylabel('绝对误差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = os.path.join(config.results_path, f"{stock_code}_rolling_prediction.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"预测图表已保存: {plot_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"绘制图表时出错: {str(e)}")

def main():
    """主滚动预测函数"""
    print("开始股票滚动价格预测...")
    print(f"配置信息:")
    print(f"  序列长度: {config.sequence_length}")
    print(f"  预测天数: {config.prediction_days}")
    print(f"  滚动窗口: {config.rolling_window}")
    print(f"  最小历史天数: {config.min_history_days}")
    print(f"  测试数据路径: {config.test_data_path}")
    print(f"  模型路径: {config.model_path}")
    print(f"  结果保存路径: {config.results_path}")
    
    # 创建结果保存目录
    os.makedirs(config.results_path, exist_ok=True)
    
    # 获取测试数据文件
    csv_files = [f for f in os.listdir(config.test_data_path) if f.endswith('.csv')]
    print(f"找到 {len(csv_files)} 个测试数据文件")
    
    # 存储所有预测结果和评估指标
    all_results = []
    all_metrics = []
    sector_performance = {}
    
    # 遍历每个股票文件进行滚动预测
    for i, csv_file in enumerate(csv_files, 1):
        stock_code = os.path.splitext(csv_file)[0]
        print(f"\n[{i}/{len(csv_files)}] 处理股票: {stock_code}")
        
        try:
            # 加载股票数据
            stock_file_path = os.path.join(config.test_data_path, csv_file)
            df = load_and_preprocess_data(stock_file_path, stock_code)
            print(f"加载数据成功，共 {len(df)} 条记录")
            
            # 确定股票所属板块
            sector = get_stock_sector(stock_code)
            print(f"股票 {stock_code} 属于板块: {sector}")
            
            # 加载对应板块的模型
            try:
                model, scaler_X, scaler_y, checkpoint = load_model_and_scalers(sector)
                print(f"成功加载板块 {sector} 的模型")
            except FileNotFoundError:
                print(f"板块 {sector} 的模型不存在，跳过股票 {stock_code}")
                continue
            
            # 进行滚动预测
            results = rolling_prediction(stock_code, model, scaler_X, scaler_y, df)
            
            if results['total_steps'] == 0:
                print(f"股票 {stock_code} 没有有效的预测结果")
                continue
            
            # 计算评估指标
            metrics = calculate_rolling_metrics(results['predictions'], results['actuals'])
            
            # 保存结果
            result_df, metrics_df = save_rolling_results(stock_code, sector, results, metrics)
            
            # 绘制预测图表
            plot_rolling_prediction(stock_code, result_df, save_plot=True)
            
            # 记录结果
            all_results.append({
                'Stock_Code': stock_code,
                'Sector': sector,
                'Success': True,
                'Total_Predictions': results['total_steps'],
                'Error_Count': len(results['errors']),
                **metrics
            })
            
            all_metrics.append(metrics_df)
            
            # 按板块统计
            if sector not in sector_performance:
                sector_performance[sector] = []
            sector_performance[sector].append(metrics)
            
            print(f"股票 {stock_code} 滚动预测完成:")
            print(f"  预测步数: {results['total_steps']}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  方向准确率: {metrics['Direction_Accuracy']:.2f}%")
            print(f"  趋势一致性: {'是' if metrics['Trend_Consistency'] else '否'}")
            
        except Exception as e:
            print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
            all_results.append({
                'Stock_Code': stock_code,
                'Sector': get_stock_sector(stock_code),
                'Success': False,
                'Error': str(e)
            })
    
    # 生成综合评估报告
    generate_rolling_evaluation_report(all_results, sector_performance, all_metrics)
    
    print(f"\n滚动预测完成！结果保存在: {config.results_path}")

def generate_rolling_evaluation_report(all_results, sector_performance, all_metrics):
    """生成滚动预测评估报告"""
    print("\n" + "="*60)
    print("生成滚动预测模型评估报告...")
    
    # 过滤成功的预测结果
    successful_results = [r for r in all_results if r.get('Success', False)]
    
    if not successful_results:
        print("没有成功的预测结果，无法生成报告")
        return
    
    # 合并所有指标数据
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        combined_metrics.to_csv(
            os.path.join(config.results_path, 'all_rolling_metrics.csv'),
            index=False, encoding='utf-8'
        )
    
    # 计算整体性能统计
    overall_metrics = {}
    metric_names = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Direction_Accuracy', 
                'Actual_Returns', 'Predicted_Returns', 'Max_Error']
    
    for metric in metric_names:
        values = [r[metric] for r in successful_results if metric in r and not np.isnan(r[metric])]
        if values:
            overall_metrics[f'{metric}_mean'] = np.mean(values)
            overall_metrics[f'{metric}_std'] = np.std(values)
            overall_metrics[f'{metric}_median'] = np.median(values)
            overall_metrics[f'{metric}_min'] = np.min(values)
            overall_metrics[f'{metric}_max'] = np.max(values)
    
    # 计算趋势一致性比例
    trend_consistency_values = [r['Trend_Consistency'] for r in successful_results if 'Trend_Consistency' in r]
    trend_consistency_rate = np.mean(trend_consistency_values) * 100 if trend_consistency_values else 0
    
    # 按板块统计性能
    sector_summary = {}
    for sector, metrics_list in sector_performance.items():
        if not metrics_list:
            continue
            
        sector_stats = {}
        for metric in metric_names:
            values = [m[metric] for m in metrics_list if metric in m and not np.isnan(m[metric])]
            if values:
                sector_stats[f'{metric}_mean'] = np.mean(values)
                sector_stats[f'{metric}_std'] = np.std(values)
                sector_stats[f'{metric}_median'] = np.median(values)
        
        # 趋势一致性统计
        trend_values = [m['Trend_Consistency'] for m in metrics_list if 'Trend_Consistency' in m]
        sector_stats['Trend_Consistency_Rate'] = np.mean(trend_values) * 100 if trend_values else 0
        sector_stats['Stock_Count'] = len(metrics_list)
        sector_stats['Total_Predictions'] = sum([m['Total_Predictions'] for m in metrics_list if 'Total_Predictions' in m])
        
        sector_summary[sector] = sector_stats
    
    # 保存结果摘要
    results_df = pd.DataFrame(successful_results)
    results_df.to_csv(os.path.join(config.results_path, 'rolling_prediction_summary.csv'), 
                    index=False, encoding='utf-8')
    
    # 生成文本报告
    report_lines = []
    report_lines.append("股票价格滚动预测模型评估报告")
    report_lines.append("=" * 60)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"预测配置:")
    report_lines.append(f"  序列长度: {config.sequence_length} 天")
    report_lines.append(f"  预测未来: {config.prediction_days} 天")
    report_lines.append(f"  滚动窗口: {config.rolling_window} 天")
    report_lines.append(f"  最小历史数据: {config.min_history_days} 天")
    report_lines.append("")
    report_lines.append(f"测试结果概览:")
    report_lines.append(f"  总测试股票数: {len(all_results)}")
    report_lines.append(f"  成功预测股票数: {len(successful_results)}")
    report_lines.append(f"  成功率: {len(successful_results)/len(all_results)*100:.2f}%")
    
    # 计算总预测步数
    total_predictions = sum([r['Total_Predictions'] for r in successful_results if 'Total_Predictions' in r])
    report_lines.append(f"  总预测步数: {total_predictions}")
    report_lines.append("")
    
    # 整体模型性能
    report_lines.append("整体模型性能:")
    report_lines.append("-" * 40)
    for metric in ['RMSE', 'MAE', 'R2', 'MAPE', 'Direction_Accuracy']:
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        median_key = f'{metric}_median'
        if mean_key in overall_metrics:
            report_lines.append(
                f"{metric:20s}: 均值={overall_metrics[mean_key]:8.4f} ± {overall_metrics[std_key]:8.4f}, "
                f"中位数={overall_metrics[median_key]:8.4f}"
            )
    
    report_lines.append(f"{'趋势一致性比例':20s}: {trend_consistency_rate:8.2f}%")
    report_lines.append("")
    
    # 收益率分析
    if 'Actual_Returns_mean' in overall_metrics and 'Predicted_Returns_mean' in overall_metrics:
        report_lines.append("收益率分析:")
        report_lines.append("-" * 40)
        report_lines.append(f"平均实际收益率: {overall_metrics['Actual_Returns_mean']:8.2f}%")
        report_lines.append(f"平均预测收益率: {overall_metrics['Predicted_Returns_mean']:8.2f}%")
        report_lines.append(f"收益率预测偏差: {abs(overall_metrics['Actual_Returns_mean'] - overall_metrics['Predicted_Returns_mean']):8.2f}%")
        report_lines.append("")
    
    # 各板块性能对比
    report_lines.append("各板块模型性能:")
    report_lines.append("-" * 40)
    
    # 按RMSE排序板块
    sector_rmse = [(sector, stats.get('RMSE_mean', float('inf'))) 
                for sector, stats in sector_summary.items()]
    sector_rmse.sort(key=lambda x: x[1])
    
    for sector, _ in sector_rmse:
        if sector not in sector_summary:
            continue
            
        stats = sector_summary[sector]
        report_lines.append(f"\n板块: {sector}")
        report_lines.append(f"  股票数量: {stats['Stock_Count']}")
        report_lines.append(f"  总预测步数: {stats.get('Total_Predictions', 0)}")
        
        for metric in ['RMSE', 'MAE', 'R2', 'MAPE', 'Direction_Accuracy']:
            mean_key = f'{metric}_mean'
            if mean_key in stats:
                report_lines.append(f"  {metric:15s}: {stats[mean_key]:8.4f}")
        
        if 'Trend_Consistency_Rate' in stats:
            report_lines.append(f"  {'趋势一致性':15s}: {stats['Trend_Consistency_Rate']:8.2f}%")
    
    # 模型表现分析
    report_lines.append(f"\n模型表现分析:")
    report_lines.append("-" * 40)
    
    # 找出表现最好和最差的股票
    if successful_results:
        best_stock = min(successful_results, key=lambda x: x.get('RMSE', float('inf')))
        worst_stock = max(successful_results, key=lambda x: x.get('RMSE', 0))
        
        report_lines.append(f"RMSE最佳股票: {best_stock['Stock_Code']} ({best_stock['Sector']}) - RMSE: {best_stock.get('RMSE', 0):.4f}")
        report_lines.append(f"RMSE最差股票: {worst_stock['Stock_Code']} ({worst_stock['Sector']}) - RMSE: {worst_stock.get('RMSE', 0):.4f}")
        
        # 方向准确率分析
        high_direction_accuracy = [r for r in successful_results if r.get('Direction_Accuracy', 0) >= 60]
        report_lines.append(f"方向准确率≥60%的股票数: {len(high_direction_accuracy)} ({len(high_direction_accuracy)/len(successful_results)*100:.1f}%)")
        
        # 趋势一致性分析
        consistent_trend = [r for r in successful_results if r.get('Trend_Consistency', False)]
        report_lines.append(f"趋势预测一致的股票数: {len(consistent_trend)} ({len(consistent_trend)/len(successful_results)*100:.1f}%)")
    
    # 性能评级
    report_lines.append(f"\n模型性能评级:")
    report_lines.append("-" * 40)
    
    # 基于MAPE的评级
    avg_mape = overall_metrics.get('MAPE_mean', float('inf'))
    if avg_mape <= 5:
        performance_grade = "优秀"
    elif avg_mape <= 10:
        performance_grade = "良好"  
    elif avg_mape <= 20:
        performance_grade = "一般"
    else:
        performance_grade = "较差"
    
    report_lines.append(f"基于MAPE的整体评级: {performance_grade} (平均MAPE: {avg_mape:.2f}%)")
    
    # 基于方向准确率的评级
    avg_direction = overall_metrics.get('Direction_Accuracy_mean', 0)
    if avg_direction >= 60:
        direction_grade = "优秀"
    elif avg_direction >= 55:
        direction_grade = "良好"
    elif avg_direction >= 50:
        direction_grade = "一般"
    else:
        direction_grade = "较差"
        
    report_lines.append(f"基于方向准确率的评级: {direction_grade} (平均方向准确率: {avg_direction:.2f}%)")
    report_lines.append("")
    
    # 改进建议
    report_lines.append("模型改进建议:")
    report_lines.append("-" * 40)
    
    if avg_mape > 15:
        report_lines.append("• MAPE较高，建议优化特征工程或调整模型结构")
    if avg_direction < 55:
        report_lines.append("• 方向预测准确率偏低，建议增加趋势相关特征")
    if trend_consistency_rate < 70:
        report_lines.append("• 长期趋势预测一致性不足，建议调整预测时间窗口")
    
    # 表现较差的板块建议
    poor_sectors = [sector for sector, stats in sector_summary.items() 
                if stats.get('RMSE_mean', 0) > overall_metrics.get('RMSE_mean', 0) * 1.2]
    if poor_sectors:
        report_lines.append(f"• 以下板块表现较差，建议针对性优化: {', '.join(poor_sectors)}")
    
    report_lines.append("")
    report_lines.append("注：滚动预测模拟真实交易场景，逐步使用新数据预测未来价格")
    
    # 保存报告
    report_file = os.path.join(config.results_path, 'rolling_evaluation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # 打印报告
    print('\n'.join(report_lines))
    
    # 保存板块统计
    if sector_summary:
        sector_df = pd.DataFrame(sector_summary).T
        sector_df.to_csv(os.path.join(config.results_path, 'rolling_sector_performance.csv'), 
                        encoding='utf-8')
    
    # 生成性能排行榜
    generate_performance_ranking(successful_results)
    
    print(f"\n详细评估报告已保存到: {config.results_path}")

def generate_performance_ranking(successful_results):
    """生成股票预测性能排行榜"""
    if not successful_results:
        return
    
    # 创建综合评分 (权重: RMSE 40%, MAPE 30%, 方向准确率 30%)
    for result in successful_results:
        rmse_score = 1 / (1 + result.get('RMSE', float('inf')))  # RMSE越小越好
        mape_score = 1 / (1 + result.get('MAPE', float('inf')))  # MAPE越小越好
        direction_score = result.get('Direction_Accuracy', 0) / 100  # 方向准确率越高越好
        
        result['Composite_Score'] = (rmse_score * 0.4 + mape_score * 0.3 + direction_score * 0.3)
    
    # 按综合评分排序
    ranked_results = sorted(successful_results, key=lambda x: x['Composite_Score'], reverse=True)
    
    # 创建排行榜DataFrame
    ranking_data = []
    for i, result in enumerate(ranked_results, 1):
        ranking_data.append({
            '排名': i,
            '股票代码': result['Stock_Code'],
            '板块': result['Sector'],
            'RMSE': result.get('RMSE', 0),
            'MAPE': result.get('MAPE', 0),
            '方向准确率': result.get('Direction_Accuracy', 0),
            '趋势一致性': '是' if result.get('Trend_Consistency', False) else '否',
            '预测步数': result.get('Total_Predictions', 0),
            '综合评分': result['Composite_Score']
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # 保存排行榜
    ranking_file = os.path.join(config.results_path, 'stock_performance_ranking.csv')
    ranking_df.to_csv(ranking_file, index=False, encoding='utf-8')
    
    # 打印前10名
    print(f"\n股票预测性能排行榜 (前10名):")
    print("-" * 80)
    print(ranking_df.head(10).to_string(index=False))
    
    print(f"\n完整排行榜已保存到: {ranking_file}")

if __name__ == "__main__":
    # 路径已经在类初始化时设置，无需重复设置
    main()