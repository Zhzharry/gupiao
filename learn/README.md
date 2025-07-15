# 训练板块 (learn)

## 板块说明
训练板块包含模型训练、预测和模型定义的核心功能，是整个系统的核心部分。现在细分为三个子模块：训练(train)、预测(predict)和模型定义(models)。

## 目录结构

```
learn/
├── train/                     # 训练模块 - 模型训练核心功能
│   ├── main.py               # 主训练程序
│   └── README.md             # 训练模块说明
│
├── predict/                   # 预测模块 - 模型预测功能
│   ├── predict.py            # 预测模块
│   └── README.md             # 预测模块说明
│
├── models/                    # 模型定义模块 - 模型架构定义
│   ├── transformer_model.py  # Transformer模型定义
│   └── README.md             # 模型定义说明
│
└── README.md                  # 训练板块说明
```

## 子模块说明

### 🎯 训练模块 (train)
负责模型训练的核心功能：
- **main.py**: 主训练程序，协调整个训练流程
- 包含数据加载、模型创建、训练循环、验证评估
- 支持GPU/CPU训练，早停、学习率调度等优化策略

### 🔮 预测模块 (predict)
负责使用训练好的模型进行预测：
- **predict.py**: 预测模块，使用训练好的模型进行股票价格预测
- 支持单步预测、多步预测、趋势预测
- 生成预测结果报告和可视化

### 🏗️ 模型定义模块 (models)
包含所有Transformer模型的架构定义：
- **transformer_model.py**: 定义各种基于Transformer的模型架构
- 包含基础Transformer、高级Transformer、LSTM-Transformer等
- 提供位置编码、多头注意力等核心组件

## 使用流程

### 1. 模型训练
```bash
cd learn/train
python main.py
```

### 2. 模型预测
```bash
cd learn/predict
python predict.py
```

## 配置参数

主要配置参数在 `train/main.py` 中设置：

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
    'patience': 15,          # 早停耐心值
    'model_type': 'basic',   # 模型类型
    'target_col': 'close',   # 预测目标
    'stock_codes': None      # 股票代码列表（None表示使用全部）
}
```

## 训练流程

1. **数据加载**: 从data板块加载处理好的数据
2. **模型创建**: 根据配置创建Transformer模型
3. **训练循环**: 执行训练、验证、早停等
4. **模型保存**: 保存最佳模型到models目录
5. **性能评估**: 计算各种评估指标

## 输出结果

- **模型文件**: 保存在 `models/` 目录
- **训练曲线**: 保存在 `../results/training_curves.png`
- **日志文件**: 保存在 `../logs/` 目录
- **预测结果**: 包含预测价格、准确率等统计信息

## 模型架构

### 基础Transformer
- 标准的Transformer编码器架构
- 适合一般的时间序列预测任务
- 包含位置编码、多头注意力、前馈网络

### 高级Transformer
- 增强的Transformer架构
- 包含时间注意力机制
- 全局平均池化
- 更复杂的输出层结构

### LSTM-Transformer
- 结合LSTM和Transformer的优势
- LSTM处理序列信息
- Transformer进行特征提取
- 特征融合输出

## 模块间依赖关系

```
train/main.py
├── 依赖 models/transformer_model.py (模型定义)
└── 依赖 data/data_processor.py (数据处理)

predict/predict.py
├── 依赖 models/transformer_model.py (模型定义)
└── 依赖 data/data_processor.py (数据处理)

models/transformer_model.py
└── 独立模块，不依赖其他模块
``` 