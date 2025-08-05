import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# 板块股票字典（新版，自动提取自用户JSON）
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

# ============================== 模型参数配置 ==============================
class ModelConfig:
    def __init__(self):
        # 数据路径配置
        self.data_path = r"data/learn_csv3"
        self.model_save_path = r"models2"
        
        # 模型架构参数
        self.sequence_length = 100  # 输入序列长度（天数）
        self.d_model = 128        # Transformer模型维度
        self.nhead = 8           # 多头注意力头数
        self.num_layers = 4      # Transformer层数
        self.dim_feedforward = 512  # 前馈网络维度
        self.dropout = 0.1       # Dropout率
        
        # 训练参数
        self.batch_size = 32     # 批次大小
        self.learning_rate = 0.001  # 学习率
        self.num_epochs = 100    # 训练轮数
        self.patience = 10       # 早停耐心值
        self.weight_decay = 1e-5 # 权重衰减
        
        # 数据处理参数
        self.test_size = 0.2     # 测试集比例
        self.random_state = 42   # 随机种子
        self.fill_method = 'forward'  # 缺失值填充方法 ('forward', 'mean', 'zero')
        
        # 预测目标（可以修改预测目标）
        self.target_column = 'close'  # 预测目标列
        self.prediction_days = 20     # 预测未来天数

config = ModelConfig()

# ============================== 数据集类 ==============================
class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# ============================== Transformer模型 ==============================
class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, output_dim=1):
        super(StockTransformer, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        
        # 使用最后一个时间步的输出
        x = x[:, -1, :]  # (batch_size, d_model)
        output = self.output_projection(x)
        return output

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

# ============================== 数据预处理函数 ==============================
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
    
    # 定义特征列（全部用英文表头）
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
            if config.fill_method == 'zero':
                df[col] = df[col].fillna(0)
            elif config.fill_method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(method='ffill').fillna(0)
    # 创建技术指标
    df = create_technical_indicators(df)
    return df

def create_technical_indicators(df):
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

def create_sequences(df, sequence_length, target_column):
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
    sequences = []
    targets = []
    for i in range(sequence_length, len(df) - config.prediction_days + 1):
        seq = df[available_features].iloc[i-sequence_length:i].values
        target = df[target_column].iloc[i + config.prediction_days - 1]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets), available_features

# ============================== 训练函数 ==============================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = model.state_dict()  # 修复：初始化，防止未赋值报错

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output.squeeze(), target).item()
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

def train_sector_model(sector_df, sector_name):
    """为单个板块训练模型"""
    print(f"\n开始训练板块 {sector_name} 的模型...")
    if len(sector_df) < config.sequence_length + config.prediction_days:
        print(f"板块 {sector_name} 数据不足，跳过训练")
        return
    # 创建序列数据
    sequences, targets, feature_names = create_sequences(sector_df, config.sequence_length, config.target_column)
    if len(sequences) == 0:
        print(f"板块 {sector_name} 无法创建有效序列，跳过训练")
        return
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
    sequences_scaled = scaler_X.fit_transform(sequences_reshaped)
    sequences_scaled = sequences_scaled.reshape(sequences.shape)
    targets_scaled = scaler_y.fit_transform(targets.reshape(-1, 1)).flatten()
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        sequences_scaled, targets_scaled, 
        test_size=config.test_size, 
        random_state=config.random_state,
        shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=config.random_state,
        shuffle=False
    )
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    input_dim = sequences.shape[-1]
    model = StockTransformer(
        input_dim=input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        output_dim=1
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        config.num_epochs, config.patience
    )
    model.eval()
    test_loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output.squeeze(), target).item()
    test_loss /= len(test_loader)
    print(f"板块 {sector_name} 测试损失: {test_loss:.6f}")
    model_dir = os.path.join(config.model_save_path, sector_name)
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'd_model': config.d_model,
            'nhead': config.nhead,
            'num_layers': config.num_layers,
            'dim_feedforward': config.dim_feedforward,
            'dropout': config.dropout
        },
        'feature_names': feature_names,
        'sequence_length': config.sequence_length,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss
    }, os.path.join(model_dir, f'{sector_name}_model.pth'))
    joblib.dump(scaler_X, os.path.join(model_dir, f'{sector_name}_scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(model_dir, f'{sector_name}_scaler_y.pkl'))
    print(f"板块 {sector_name} 模型训练完成并保存")


def main():
    """主训练函数（分板块训练，每板块一个模型）"""
    os.makedirs(config.model_save_path, exist_ok=True)
    csv_files = [f for f in os.listdir(config.data_path) if f.endswith('.csv')]
    csv_codes = {os.path.splitext(f)[0]: f for f in csv_files}
    print(f"找到 {len(csv_files)} 个股票数据文件")
    print(f"模型配置:")
    print(f"  序列长度: {config.sequence_length}")
    print(f"  模型维度: {config.d_model}")
    print(f"  注意力头数: {config.nhead}")
    print(f"  层数: {config.num_layers}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  训练轮数: {config.num_epochs}")
    print(f"  预测目标: {config.target_column}")
    for sector, code_list in 板块股票.items():
        print(f"\n=== 开始训练板块：{sector} ===")
        dfs = []
        for code in code_list:
            if code in csv_codes:
                csv_file = csv_codes[code]
                stock_file_path = os.path.join(config.data_path, csv_file)
                try:
                    df = load_and_preprocess_data(stock_file_path, code)
                    df['stock_code'] = code  # 可选：加一列标记股票代码
                    dfs.append(df)
                except Exception as e:
                    print(f"加载股票 {csv_file} 时发生错误: {str(e)}")
            else:
                print(f"股票 {code} 数据文件缺失，跳过")
        if not dfs:
            print(f"板块 {sector} 无可用数据，跳过")
            continue
        sector_df = pd.concat(dfs, ignore_index=True)
        train_sector_model(sector_df, sector)
    print("\n全部板块训练完成！")
    print(f"模型保存路径: {config.model_save_path}")

if __name__ == "__main__":
    main()