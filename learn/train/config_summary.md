 🎯 股票Transformer训练模型 - 配置参数总结

## 📋 快速配置指南

### 🔧 最常用的配置修改

#### 1. 预测周期调整
```python
self.prediction_period = 180  # 预测未来天数
# 建议值：
# - 短期预测：7-30天
# - 中期预测：30-90天  
# - 长期预测：90-180天
```

#### 2. 分类策略选择
```python
self.classification_method = "nine_class"  # 分类方法
self.num_classes = 9  # 分类数量
# 可选策略：
# - "nine_class": 九分类法，适合多等级预测
# - "percentile": 百分位分类，适合相对排名
# - "threshold": 阈值分类，适合绝对收益
```

#### 3. 模型大小调整
```python
self.d_model = 128      # Transformer维度
self.nhead = 8          # 注意力头数
self.num_layers = 4     # Transformer层数
# 建议配置：
# - 小模型：d_model=64, nhead=4, num_layers=2
# - 中模型：d_model=128, nhead=8, num_layers=4
# - 大模型：d_model=256, nhead=16, num_layers=6
```

#### 4. 训练参数优化
```python
self.batch_size = 32        # 批次大小
self.learning_rate = 0.001  # 学习率
self.num_epochs = 50        # 训练轮数
# 建议值：
# - 小数据集：batch_size=16, learning_rate=0.003
# - 大数据集：batch_size=64, learning_rate=0.001
```

## 📊 完整配置参数列表

### 🗂️ 数据路径配置
```python
self.data_dir = r"D:\programming\Workspace\gupiao\learn\train\split_data\train"
self.model_save_path = "stock_transformer_model.pth"
```

### 🔄 数据预处理配置
```python
self.seq_len = 60              # 输入序列长度（30-120）
self.prediction_period = 180    # 预测未来天数（7-365）
```

### ⚙️ 特征工程配置
```python
# 特征开关
self.use_price_features = True      # 价格特征
self.use_volume_features = True     # 成交量特征
self.use_technical_indicators = True # 技术指标

# 技术指标参数
self.ma_periods = [5, 10, 20, 60]  # 移动平均线周期
self.rsi_period = 14               # RSI周期
self.volatility_periods = [5, 20]  # 波动率计算周期
```

### 🎯 分类策略配置
```python
# 分类方法
self.classification_method = "nine_class"  # 九分类
self.num_classes = 9                       # 分类数量

# 百分位分类参数
self.positive_percentile = 20  # 正样本百分位
self.negative_percentile = 20  # 负样本百分位

# 阈值分类参数
self.positive_threshold = 5.0  # 正样本涨幅阈值
self.negative_threshold = 0.0  # 负样本涨幅阈值
```

### 🏗️ 模型架构配置
```python
self.d_model = 128      # Transformer维度（64-256）
self.nhead = 8          # 注意力头数（4-16）
self.num_layers = 4     # Transformer层数（2-8）
self.dropout = 0.1      # Dropout率（0.05-0.3）
```

### 🚀 训练配置
```python
# 基础训练参数
self.batch_size = 32        # 批次大小（16-128）
self.learning_rate = 0.001  # 学习率（0.0001-0.01）
self.num_epochs = 50        # 训练轮数（30-200）
self.weight_decay = 0.01    # 权重衰减（0.001-0.1）

# 学习率调度
self.scheduler_step_size = 10  # 调度步长
self.scheduler_gamma = 0.8     # 衰减因子

# 数据划分
self.test_size = 0.2      # 测试集比例（0.1-0.3）
self.random_state = 42     # 随机种子
```

### 📈 评估配置
```python
self.eval_frequency = 10   # 评估频率
self.save_model = True     # 保存模型
self.plot_results = True   # 绘制结果
```

## 🎯 推荐配置组合

### 1. 短期预测配置
```python
self.prediction_period = 30
self.classification_method = "threshold"
self.num_classes = 3
self.seq_len = 30
self.batch_size = 64
self.learning_rate = 0.003
```

### 2. 中期预测配置
```python
self.prediction_period = 90
self.classification_method = "percentile"
self.num_classes = 3
self.seq_len = 60
self.batch_size = 32
self.learning_rate = 0.001
```

### 3. 长期预测配置
```python
self.prediction_period = 180
self.classification_method = "nine_class"
self.num_classes = 9
self.seq_len = 90
self.batch_size = 16
self.learning_rate = 0.0005
```

### 4. 高性能配置
```python
self.d_model = 256
self.nhead = 16
self.num_layers = 6
self.batch_size = 16
self.learning_rate = 0.0001
self.num_epochs = 100
```

## 🔍 参数调优建议

### 1. 如果模型过拟合
- 增加 `dropout` 到 0.2-0.3
- 增加 `weight_decay` 到 0.05-0.1
- 减少 `num_layers` 或 `d_model`
- 增加 `batch_size`

### 2. 如果模型欠拟合
- 减少 `dropout` 到 0.05-0.1
- 减少 `weight_decay` 到 0.001-0.01
- 增加 `num_layers` 或 `d_model`
- 增加 `num_epochs`

### 3. 如果训练速度慢
- 减少 `batch_size`
- 减少 `seq_len`
- 减少 `d_model` 或 `num_layers`
- 使用GPU训练

### 4. 如果内存不足
- 减少 `batch_size`
- 减少 `seq_len`
- 减少 `d_model`
- 关闭不必要的特征

## 📝 使用说明

1. **修改配置**：在 `ModelConfig` 类中修改相应参数
2. **运行训练**：直接运行脚本开始训练
3. **监控训练**：查看控制台输出的训练进度
4. **查看结果**：训练完成后查看保存的模型和图表

## 🎯 核心改进策略

### 1. 长期预测策略
- 从预测"明天涨跌"改为预测"中长期表现"
- 关注年度大牛股（一年内涨5倍以上的股票）

### 2. 样本分类策略
- **九分类法**：将股票分为9个等级，只在最优类别时出手
- **百分位分类**：取前20%作为正样本，底部20%作为负样本
- **阈值分类**：涨幅≥5倍为正样本，涨幅≤0为负样本

### 3. 多分类方法
- 支持3分类、9分类等多种分类方式
- 只在最优类别时发出买入信号
- 其他风险类别不操作，降低误判
