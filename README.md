# 股票价格预测模型 - 基于Transformer的深度学习系统

## 项目概述

这是一个基于Transformer架构的股票价格预测系统，使用深度学习技术分析历史股票数据，预测未来价格走势。

### 项目特点
- 🚀 **先进的Transformer架构**：使用多头注意力机制捕捉股票价格的时间序列模式
- 📊 **丰富的技术指标**：集成移动平均、MACD、RSI、布林带等20+技术指标
- 🎯 **多股票支持**：支持同时训练和预测多只股票
- 📈 **完整的训练流程**：包含数据预处理、模型训练、验证评估的完整流程
- 🔧 **灵活的配置**：支持多种模型配置和参数调整

## 项目结构

```
learn_transformer/
├── data/                      # 数据板块 - 数据处理和特征工程
│   ├── data_processor.py     # 核心数据处理模块
│   ├── check_data.py         # 数据检查工具
│   ├── README.md             # 数据板块说明
│   ├── learn_csv/            # 训练数据目录
│   └── test_csv/             # 测试数据目录
│
├── learn/                     # 训练板块 - 模型训练和预测
│   ├── train/                # 训练模块 - 模型训练核心功能
│   │   ├── main.py           # 主训练程序
│   │   └── README.md         # 训练模块说明
│   ├── predict/              # 预测模块 - 模型预测功能
│   │   ├── predict.py        # 预测模块
│   │   └── README.md         # 预测模块说明
│   ├── models/               # 模型定义模块 - 模型架构定义
│   │   ├── transformer_model.py # Transformer模型定义
│   │   └── README.md         # 模型定义说明
│   └── README.md             # 训练板块说明
│
├── test/                      # 测试板块 - 功能验证和测试
│   ├── test_data.py          # 数据测试模块
│   └── README.md             # 测试板块说明
│
├── before_learn/              # 配置板块 - 环境配置和依赖安装
│   ├── install_requirements.py # 依赖安装工具
│   └── README.md             # 配置板块说明
│
├── models/                    # 模型保存目录
├── results/                   # 结果输出目录
├── logs/                      # 日志文件目录
└── README.md                  # 项目说明文档
```

## 板块说明

### 📊 数据板块 (data)
负责所有与数据处理相关的功能：
- **data_processor.py**: 核心数据处理模块，包含数据加载、清洗、技术指标计算
- **check_data.py**: 数据检查工具，验证数据完整性和格式

### 🎯 训练板块 (learn)
包含模型训练、预测和模型定义的核心功能，细分为三个子模块：

#### 🎯 训练模块 (learn/train)
- **main.py**: 主训练程序，协调整个训练流程
- 包含数据加载、模型创建、训练循环、验证评估

#### 🔮 预测模块 (learn/predict)
- **predict.py**: 预测模块，使用训练好的模型进行股票价格预测
- 支持单步预测、多步预测、趋势预测

#### 🏗️ 模型定义模块 (learn/models)
- **transformer_model.py**: Transformer模型定义，包含多种模型架构
- 提供位置编码、多头注意力等核心组件

### 🧪 测试板块 (test)
负责验证和测试系统的各个组件：
- **test_data.py**: 数据测试模块，验证数据处理功能的正确性

### ⚙️ 配置板块 (before_learn)
包含系统运行前的准备工作：
- **install_requirements.py**: 依赖安装工具，自动安装所需依赖包

## 快速开始

### 1. 环境准备
```bash
cd before_learn
python install_requirements.py
```

### 2. 数据检查
```bash
cd data
python check_data.py
```

### 3. 数据测试
```bash
cd test
python test_data.py
```

### 4. 模型训练
```bash
cd learn/train
python main.py
```

### 5. 价格预测
```bash
cd learn/predict
python predict.py
```

## 技术架构

### 模型架构
- **基础Transformer**: 适合一般预测任务的基础模型
- **高级Transformer**: 包含更多特性的增强模型
- **LSTM-Transformer**: LSTM和Transformer的混合模型

### 技术指标
- 移动平均线（SMA 5/10/20）
- 指数移动平均线（EMA 12/26）
- MACD指标（MACD、信号线、柱状图）
- RSI相对强弱指标
- 布林带（上轨、下轨、中轨）
- 成交量比率和价格位置指标
- 波动率指标

## 配置参数

主要配置参数在 `learn/train/main.py` 中设置：

```python
config = {
    'seq_length': 30,        # 序列长度
    'input_dim': 21,         # 输入特征维度
    'd_model': 96,           # 模型维度
    'nhead': 8,              # 注意力头数
    'num_layers': 3,         # Transformer层数
    'batch_size': 16,        # 批次大小
    'learning_rate': 0.001,  # 学习率
    'epochs': 50,            # 训练轮数
    'stock_codes': None      # 股票代码列表（None表示使用全部）
}
```

## 输出结果

- **模型文件**: 保存在 `models/` 目录（项目根目录）
- **训练曲线**: 保存在 `results/training_curves.png`（项目根目录）
- **预测结果**: 包含预测价格、准确率统计等
- **日志文件**: 保存在 `logs/` 目录

## 系统要求

- Python 3.8+
- PyTorch 1.9+
- CUDA支持（可选，用于GPU加速）
- 至少8GB内存（推荐16GB+）

## 使用流程

1. **配置环境** → 进入 `before_learn` 板块安装依赖
2. **检查数据** → 进入 `data` 板块验证数据完整性
3. **测试功能** → 进入 `test` 板块验证系统功能
4. **训练模型** → 进入 `learn` 板块进行模型训练
5. **进行预测** → 使用训练好的模型进行价格预测

## 作者

AI Assistant  
创建时间：2024年

## 许可证

本项目仅供学习和研究使用。 