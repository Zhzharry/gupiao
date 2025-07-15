"""
股票价格预测模型 - Transformer模型定义
========================================

文件作用：
- 定义各种基于Transformer的股票价格预测模型
- 包含基础Transformer、高级Transformer、LSTM-Transformer等模型架构
- 提供位置编码、多头注意力机制等核心组件
- 支持不同的模型配置和参数设置

主要模型：
1. StockTransformer: 基础Transformer模型，适合一般预测任务
2. AdvancedStockTransformer: 高级Transformer模型，包含更多特性
3. LSTMTransformer: LSTM+Transformer混合模型
4. PositionalEncoding: 位置编码模块

核心功能：
- 多头自注意力机制
- 位置编码
- 残差连接和层归一化
- 前馈神经网络
- 时间序列特征提取

使用方法：
- 通过create_model()函数创建模型
- 支持不同的模型类型和参数配置

作者：AI Assistant
创建时间：2024年
"""

# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
import math  # 数学函数
import numpy as np  # 数值计算库

class PositionalEncoding(nn.Module):
    pe: torch.Tensor  # 类型注解，修复类型检查报错
    """位置编码模块，为Transformer提供序列位置信息"""
    
    def __init__(self, d_model, max_len=5000):
        """初始化位置编码
        
        Args:
            d_model (int): 模型维度
            max_len (int): 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置索引
        
        # 计算位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))  # 分母项
        
        # 正弦和余弦编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # 调整维度
        self.register_buffer('pe', pe)  # 注册为缓冲区（不参与梯度更新）
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [seq_len, batch_size, d_model]
            
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        return x + self.pe[:x.size(0), :]  # 添加位置编码

class StockTransformer(nn.Module):
    """基础股票Transformer模型"""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len, output_dim=1, dropout=0.1):
        """初始化基础Transformer模型
        
        Args:
            input_dim (int): 输入特征维度
            d_model (int): 模型维度
            nhead (int): 注意力头数
            num_layers (int): Transformer层数
            seq_len (int): 序列长度
            output_dim (int): 输出维度
            dropout (float): Dropout率
        """
        super(StockTransformer, self).__init__()
        
        self.input_dim = input_dim  # 保存输入维度
        self.d_model = d_model  # 保存模型维度
        self.seq_len = seq_len  # 保存序列长度
        
        # 输入投影层（将输入特征映射到模型维度）
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 模型维度
            nhead=nhead,  # 注意力头数
            dim_feedforward=d_model * 4,  # 前馈网络维度
            dropout=dropout,  # Dropout率
            batch_first=True  # 批次维度在前
        )
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers  # 编码器层数
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 全连接层1
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # Dropout
            nn.Linear(d_model // 2, output_dim)  # 全连接层2
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 预测结果，形状为 [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape  # 获取输入形状
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)  # 添加位置编码
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 应用Dropout
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # 输出层
        output = self.output_layer(x)  # [batch_size, output_dim]
        
        return output

class AdvancedStockTransformer(nn.Module):
    """高级股票Transformer模型，包含更多特性"""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len, output_dim=1, dropout=0.1):
        """初始化高级Transformer模型
        
        Args:
            input_dim (int): 输入特征维度
            d_model (int): 模型维度
            nhead (int): 注意力头数
            num_layers (int): Transformer层数
            seq_len (int): 序列长度
            output_dim (int): 输出维度
            dropout (float): Dropout率
        """
        super(AdvancedStockTransformer, self).__init__()
        
        self.input_dim = input_dim  # 保存输入维度
        self.d_model = d_model  # 保存模型维度
        self.seq_len = seq_len  # 保存序列长度
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, 
                nhead, 
                dropout=dropout, 
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 前馈网络层
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),  # 扩展维度
                nn.ReLU(),  # 激活函数
                nn.Dropout(dropout),  # Dropout
                nn.Linear(d_model * 4, d_model)  # 压缩维度
            ) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # 时间注意力层（关注时间序列模式）
        self.temporal_attention = nn.MultiheadAttention(
            d_model, 
            nhead, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层（更复杂的结构）
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),  # 全连接层1
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # Dropout
            nn.Linear(d_model, d_model // 2),  # 全连接层2
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # Dropout
            nn.Linear(d_model // 2, output_dim)  # 输出层
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 预测结果，形状为 [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape  # 获取输入形状
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)  # 添加位置编码
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 应用Dropout
        x = self.dropout(x)
        
        # 多层Transformer编码
        for i in range(len(self.attention_layers)):
            # 多头自注意力
            attn_output, _ = self.attention_layers[i](x, x, x)  # 自注意力
            x = self.norm1_layers[i](x + attn_output)  # 残差连接和层归一化
            
            # 前馈网络
            ff_output = self.feed_forward_layers[i](x)  # 前馈网络
            x = self.norm2_layers[i](x + ff_output)  # 残差连接和层归一化
        
        # 时间注意力（关注整个序列）
        temporal_output, _ = self.temporal_attention(x, x, x)  # 时间注意力
        x = x + temporal_output  # 残差连接
        
        # 全局平均池化
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch_size, d_model]
        
        # 输出层
        output = self.output_layer(x)  # [batch_size, output_dim]
        
        return output

class LSTMTransformer(nn.Module):
    """LSTM + Transformer混合模型"""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len, output_dim=1, dropout=0.1):
        """初始化LSTM+Transformer混合模型
        
        Args:
            input_dim (int): 输入特征维度
            d_model (int): 模型维度
            nhead (int): 注意力头数
            num_layers (int): Transformer层数
            seq_len (int): 序列长度
            output_dim (int): 输出维度
            dropout (float): Dropout率
        """
        super(LSTMTransformer, self).__init__()
        
        self.input_dim = input_dim  # 保存输入维度
        self.d_model = d_model  # 保存模型维度
        self.seq_len = seq_len  # 保存序列长度
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,  # 输入维度
            hidden_size=d_model,  # 隐藏层维度
            num_layers=2,  # LSTM层数
            dropout=dropout,  # Dropout率
            batch_first=True  # 批次维度在前
        )
        
        # 输入投影层（将输入特征映射到模型维度）
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 模型维度
            nhead=nhead,  # 注意力头数
            dim_feedforward=d_model * 4,  # 前馈网络维度
            dropout=dropout,  # Dropout率
            batch_first=True  # 批次维度在前
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers  # 编码器层数
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 全连接层1（LSTM + Transformer特征）
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # Dropout
            nn.Linear(d_model, output_dim)  # 输出层
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 预测结果，形状为 [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape  # 获取输入形状
        
        # LSTM处理
        lstm_output, (hidden, cell) = self.lstm(x)  # LSTM前向传播
        lstm_features = lstm_output[:, -1, :]  # 取最后一个时间步的LSTM输出
        
        # 输入投影（为Transformer准备）
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)  # 添加位置编码
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 应用Dropout
        x = self.dropout(x)
        
        # Transformer编码
        transformer_output = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        transformer_features = transformer_output[:, -1, :]  # 取最后一个时间步的Transformer输出
        
        # 特征融合
        combined_features = torch.cat([lstm_features, transformer_features], dim=1)  # 拼接特征
        
        # 输出层
        output = self.output_layer(combined_features)  # [batch_size, output_dim]
        
        return output

def create_model(model_type, **kwargs):
    """创建指定类型的模型
    
    Args:
        model_type (str): 模型类型 ('basic', 'advanced', 'lstm_transformer')
        **kwargs: 模型参数
        
    Returns:
        nn.Module: 创建的模型实例
    """
    if model_type == 'basic':  # 基础Transformer模型
        return StockTransformer(**kwargs)
    elif model_type == 'advanced':  # 高级Transformer模型
        return AdvancedStockTransformer(**kwargs)
    elif model_type == 'lstm_transformer':  # LSTM+Transformer混合模型
        return LSTMTransformer(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# 模型测试函数
def test_model():
    """测试模型功能"""
    # 测试参数
    batch_size = 4  # 批次大小
    seq_len = 60  # 序列长度
    input_dim = 21  # 输入维度
    d_model = 256  # 模型维度
    nhead = 8  # 注意力头数
    num_layers = 6  # 层数
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, input_dim)  # 随机测试数据
    
    # 测试基础Transformer
    print("测试基础Transformer模型...")
    basic_model = StockTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len
    )
    output = basic_model(x)
    print(f"基础Transformer输出形状: {output.shape}")
    
    # 测试高级Transformer
    print("测试高级Transformer模型...")
    advanced_model = AdvancedStockTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len
    )
    output = advanced_model(x)
    print(f"高级Transformer输出形状: {output.shape}")
    
    # 测试LSTM+Transformer
    print("测试LSTM+Transformer模型...")
    lstm_transformer_model = LSTMTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len
    )
    output = lstm_transformer_model(x)
    print(f"LSTM+Transformer输出形状: {output.shape}")
    
    print("所有模型测试通过！")

if __name__ == "__main__":
    test_model()  # 运行模型测试 