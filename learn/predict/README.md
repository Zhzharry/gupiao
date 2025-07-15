# 预测模块 (predict)

## 模块说明
预测模块负责使用训练好的模型进行股票价格预测，包括模型加载、数据预处理、预测执行和结果分析。

## 文件说明

### predict.py
**作用**: 预测模块，使用训练好的模型进行股票价格预测
- 加载训练好的Transformer模型
- 对预测数据进行预处理
- 执行单步和多步预测
- 生成预测结果报告和可视化

**主要功能**:
- `StockPredictor`类：负责预测功能
- `load_model()`: 加载训练好的模型
- `prepare_prediction_data()`: 准备预测数据
- `predict_single_step()`: 单步预测
- `predict_multi_step()`: 多步预测
- `analyze_results()`: 分析预测结果

**使用方法**:
```bash
cd learn/predict
python predict.py
```

## 预测功能

### 1. 单步预测
- 预测下一个时间点的股票价格
- 基于历史序列数据
- 输出预测价格和置信区间

### 2. 多步预测
- 预测未来多个时间点的价格
- 支持滚动预测和批量预测
- 生成价格趋势预测

### 3. 趋势预测
- 预测价格变化趋势
- 判断上涨、下跌或横盘
- 提供趋势置信度

### 4. 置信区间
- 计算预测的置信区间
- 提供预测的不确定性评估
- 支持不同置信水平

## 预测流程

### 1. 模型加载
- 加载训练好的模型文件
- 验证模型完整性
- 设置预测参数

### 2. 数据预处理
- 加载最新的股票数据
- 计算技术指标
- 数据标准化处理

### 3. 预测执行
- 执行模型推理
- 处理预测结果
- 计算置信区间

### 4. 结果分析
- 生成预测报告
- 创建可视化图表
- 保存预测结果

## 输出结果

### 预测报告
- 预测价格和置信区间
- 预测准确率统计
- 趋势分析结果
- 风险评估信息

### 可视化图表
- 历史价格vs预测价格
- 预测趋势图
- 置信区间图
- 误差分析图

### 数据文件
- 预测结果CSV文件
- 预测报告JSON文件
- 可视化图表PNG文件

## 预测配置

### 模型配置
```python
config = {
    'model_path': '../models/best_model.pth',  # 模型文件路径（项目根目录的models文件夹）
    'seq_length': 30,                          # 序列长度
    'input_dim': 21,                           # 输入特征维度
    'model_type': 'basic',                     # 模型类型
    'device': 'cuda'                           # 计算设备
}
```

### 预测参数
```python
prediction_config = {
    'confidence_level': 0.95,    # 置信水平
    'prediction_steps': 5,       # 预测步数
    'rolling_window': True,      # 是否使用滚动窗口
    'output_format': 'json'      # 输出格式
}
```

## 使用示例

### 单只股票预测
```python
predictor = StockPredictor(config)
result = predictor.predict_single_stock('000001')
print(f"预测价格: {result['predicted_price']}")
print(f"置信区间: {result['confidence_interval']}")
```

### 多只股票预测
```python
stocks = ['000001', '000002', '000003']
results = predictor.predict_multiple_stocks(stocks)
for stock, result in results.items():
    print(f"{stock}: {result['predicted_price']}")
```

## 注意事项

### 数据要求
- 确保预测数据格式与训练数据一致
- 检查数据完整性和时效性
- 验证技术指标计算的正确性

### 模型要求
- 确保模型文件存在且完整
- 验证模型配置与训练时一致
- 检查模型版本兼容性

### 性能优化
- 使用GPU加速预测
- 批量处理提高效率
- 缓存中间结果减少计算 