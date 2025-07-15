# 模型定义模块 (models)

## 模块说明
模型定义模块包含所有Transformer模型的架构定义，提供多种模型类型供训练和预测使用。

## 文件说明

### transformer_model.py
**作用**: Transformer模型定义文件
- 定义各种基于Transformer的模型架构
- 包含位置编码、多头注意力等核心组件
- 支持不同的模型配置和参数设置

**主要模型**:
- `StockTransformer`: 基础Transformer模型
- `AdvancedStockTransformer`: 高级Transformer模型
- `LSTMTransformer`: LSTM+Transformer混合模型
- `PositionalEncoding`: 位置编码模块

**核心功能**:
- 多头自注意力机制
- 位置编码
- 残差连接和层归一化
- 前馈神经网络
- 时间序列特征提取

## 模型架构

### 1. 基础Transformer (StockTransformer)
**特点**: 标准的Transformer编码器架构
- 适合一般的时间序列预测任务
- 包含位置编码、多头注意力、前馈网络
- 结构简单，训练稳定

**参数配置**:
```python
model = StockTransformer(
    input_dim=21,      # 输入特征维度
    d_model=96,        # 模型维度
    nhead=8,           # 注意力头数
    num_layers=3,      # Transformer层数
    seq_len=30,        # 序列长度
    output_dim=1,      # 输出维度
    dropout=0.1        # Dropout率
)
```

### 2. 高级Transformer (AdvancedStockTransformer)
**特点**: 增强的Transformer架构
- 包含时间注意力机制
- 全局平均池化
- 更复杂的输出层结构
- 更好的特征提取能力

**增强功能**:
- 时间注意力层
- 多层输出结构
- 特征融合机制
- 自适应池化

### 3. LSTM-Transformer (LSTMTransformer)
**特点**: LSTM和Transformer的混合模型
- 结合LSTM和Transformer的优势
- LSTM处理序列信息
- Transformer进行特征提取
- 特征融合输出

**混合架构**:
- LSTM编码器
- Transformer特征提取器
- 特征融合层
- 输出预测层

## 核心组件

### PositionalEncoding
**作用**: 位置编码模块
- 为Transformer提供序列位置信息
- 使用正弦余弦编码
- 支持不同长度的序列

**实现特点**:
- 可学习的位置编码
- 支持长序列
- 位置信息保持

### 多头注意力机制
**作用**: 核心注意力组件
- 并行计算多个注意力头
- 捕捉不同的特征关系
- 提高模型表达能力

**参数设置**:
- 注意力头数：通常为8或16
- 模型维度：必须能被头数整除
- 缩放因子：防止梯度消失

## 模型选择指南

### 选择基础Transformer的情况
- 数据量相对较小
- 计算资源有限
- 需要快速原型验证
- 对模型复杂度要求不高

### 选择高级Transformer的情况
- 数据量充足
- 有足够的计算资源
- 需要更好的预测性能
- 对模型表达能力要求高

### 选择LSTM-Transformer的情况
- 序列依赖性强
- 需要捕捉长期依赖
- 数据具有明显的时序特征
- 传统LSTM效果不佳

## 模型配置

### 通用配置参数
```python
config = {
    'input_dim': 21,        # 输入特征维度
    'd_model': 96,          # 模型维度
    'nhead': 8,             # 注意力头数
    'num_layers': 3,        # 层数
    'seq_len': 30,          # 序列长度
    'output_dim': 1,        # 输出维度
    'dropout': 0.1,         # Dropout率
    'model_type': 'basic'   # 模型类型
}
```

### 性能优化建议
- **模型维度**: 根据数据复杂度调整
- **注意力头数**: 通常为8或16
- **层数**: 根据任务复杂度选择
- **Dropout**: 防止过拟合

## 扩展开发

### 添加新模型
1. 继承基础模型类
2. 实现forward方法
3. 添加模型配置
4. 更新create_model函数

### 自定义组件
1. 定义新的注意力机制
2. 实现自定义位置编码
3. 添加新的特征提取层
4. 集成外部预训练模型 