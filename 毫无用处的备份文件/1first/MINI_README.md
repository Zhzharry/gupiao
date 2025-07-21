# Mini Training Script (mini.py)

## 概述

`mini.py` 是一个专门用于快速可行性验证的小规模训练脚本。它使用少量数据和简化的模型配置来快速验证Transformer模型在股票预测任务上的可行性。

## 特点

- **快速验证**: 使用少量数据（最多5只股票，30天数据）
- **简化模型**: 较小的Transformer模型（64维，2层，4头）
- **快速训练**: 仅5个epoch，小批量大小
- **完整流程**: 包含数据加载、训练、验证、测试和可视化

## 配置参数

```python
mini_config = {
    'sequence_length': 10,    # 输入序列长度
    'prediction_length': 3,   # 预测长度
    'batch_size': 8,          # 批量大小
    'epochs': 5,              # 训练轮数
    'learning_rate': 0.001,   # 学习率
    'd_model': 64,            # 模型维度
    'nhead': 4,               # 注意力头数
    'num_layers': 2,          # Transformer层数
    'dropout': 0.1,           # Dropout率
    'max_stocks': 5,          # 最大股票数量
    'max_days': 30,           # 最大天数
    'train_ratio': 0.8,       # 训练集比例
    'validation_ratio': 0.1,  # 验证集比例
    'test_ratio': 0.1         # 测试集比例
}
```

## 使用方法

### 1. 直接运行

```bash
cd learn/train
python mini.py
```

### 2. 在Python中调用

```python
from mini import MiniTrainer

# 创建配置
class Config:
    pass
config = Config()

# 创建训练器
trainer = MiniTrainer(config)

# 加载数据
train_data, val_data, test_data = trainer.load_mini_data()

# 创建模型
trainer.create_model()

# 训练
train_losses, val_losses = trainer.train_mini(train_data, val_data)

# 测试
mse, mae, rmse = trainer.test_mini_model(test_data)
```

## 输出文件

### 模型文件
- `models/mini_model.pth`: 训练好的mini模型

### 结果文件
- `results/mini_predictions.png`: 预测结果可视化图

## 数据要求

- 数据文件应位于 `data/` 目录下
- 支持CSV格式的股票数据文件
- 每个文件应包含以下列：`date`, `stock`, `open`, `high`, `low`, `close`, `volume`

## 性能评估

脚本会输出以下指标：
- **MSE (Mean Squared Error)**: 均方误差
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **RMSE (Root Mean Squared Error)**: 均方根误差

## 预期结果

- **训练时间**: 通常1-3分钟
- **模型参数**: 约10K-50K参数
- **内存使用**: 低（适合CPU训练）
- **验证结果**: 快速获得可行性评估

## 故障排除

### 常见问题

1. **无数据文件**
   ```
   No CSV files found in data directory!
   ```
   解决：确保 `data/` 目录下有CSV文件

2. **CUDA内存不足**
   ```
   CUDA out of memory
   ```
   解决：脚本会自动使用CPU，或减少 `batch_size`

3. **模型参数错误**
   ```
   d_model must be divisible by nhead
   ```
   解决：确保 `d_model` 能被 `nhead` 整除

### 调试建议

1. 检查数据文件格式
2. 验证数据文件路径
3. 确认依赖包已安装
4. 查看控制台输出的详细信息

## 扩展使用

### 修改配置

可以通过修改 `mini_config` 字典来调整参数：

```python
# 在MiniTrainer类中修改
self.mini_config['epochs'] = 10  # 增加训练轮数
self.mini_config['max_stocks'] = 10  # 增加股票数量
self.mini_config['d_model'] = 128  # 增加模型复杂度
```

### 自定义评估

可以添加自定义的评估指标：

```python
def custom_evaluation(self, y_true, y_pred):
    # 添加自定义评估逻辑
    directional_accuracy = self.calculate_directional_accuracy(y_true, y_pred)
    return directional_accuracy
```

## 与完整训练的关系

mini.py 是完整训练流程的简化版本：
- 使用相同的模型架构
- 使用相同的数据处理流程
- 使用相同的评估指标
- 但使用更小的数据集和更简单的配置

成功运行mini.py后，可以：
1. 验证模型架构的可行性
2. 检查数据处理流程的正确性
3. 评估计算资源需求
4. 为完整训练提供参数参考 