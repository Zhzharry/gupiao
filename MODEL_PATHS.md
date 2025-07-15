# 模型保存路径配置说明

## 概述

本项目已正确配置模型保存路径，确保训练好的模型文件保存在项目根目录的 `models/` 文件夹下。

## 路径配置      hgfhjgjhg

### 训练模块 (learn/train/main.py)

当从 `learn/train/` 目录运行训练脚本时：

```python
# 模型保存路径
torch.save(model.state_dict(), f"../models/best_model.pth")

# 训练曲线保存路径
plt.savefig('../results/training_curves.png', dpi=300, bbox_inches='tight')

# 数据目录路径
train_data_dir = "../data/learn_csv"
test_data_dir = "../data/test_csv"
```

### 预测模块 (learn/predict/predict.py)

当从 `learn/predict/` 目录运行预测脚本时：

```python
# 模型加载路径
model_path = "../models/best_model.pth"

# 数据目录路径
data_dir = "../data/test_csv"

# 预测结果保存路径
plt.savefig(f'../results/prediction_{stock_code}.png', dpi=300, bbox_inches='tight')
```

## 目录结构

```
learn_transformer/
├── models/                    # 模型保存目录（项目根目录）
│   └── best_model.pth        # 训练好的最佳模型
├── results/                   # 结果输出目录
│   ├── training_curves.png   # 训练曲线
│   └── prediction_*.png      # 预测结果图
├── logs/                      # 日志文件目录
├── data/                      # 数据目录
│   ├── learn_csv/            # 训练数据
│   └── test_csv/             # 测试数据
└── learn/                     # 训练板块
    ├── train/                # 训练模块
    └── predict/              # 预测模块
```

## 使用方法

### 1. 训练模型

```bash
cd learn/train
python main.py
```

训练完成后，模型将保存在 `../models/best_model.pth`

### 2. 使用模型预测

```bash
cd learn/predict
python predict.py
```

预测脚本将从 `../models/best_model.pth` 加载模型

## 路径验证

运行路径测试脚本验证配置：

```bash
python test_paths.py
```

## 注意事项

1. **相对路径**: 所有路径都使用相对路径，确保在不同环境下都能正常工作
2. **目录创建**: 训练脚本会自动创建必要的目录（models、results、logs）
3. **路径一致性**: 训练和预测模块使用相同的路径配置，确保模型能正确加载
4. **跨平台兼容**: 使用 `os.path.join()` 和相对路径，确保在Windows、Linux、macOS上都能正常工作

## 故障排除

### 问题1: 模型文件找不到
**原因**: 路径配置错误或模型未训练
**解决**: 
- 检查是否已运行训练脚本
- 验证 `models/` 目录是否存在
- 确认模型文件 `best_model.pth` 是否存在

### 问题2: 数据目录找不到
**原因**: 数据目录路径错误
**解决**:
- 检查 `data/learn_csv/` 和 `data/test_csv/` 目录是否存在
- 确认数据文件格式正确

### 问题3: 结果目录权限错误
**原因**: 目录权限不足
**解决**:
- 检查目录写入权限
- 手动创建 `results/` 目录

## 配置检查清单

- [ ] `models/` 目录存在且可写
- [ ] `results/` 目录存在且可写
- [ ] `logs/` 目录存在且可写
- [ ] 训练数据目录 `data/learn_csv/` 存在
- [ ] 测试数据目录 `data/test_csv/` 存在
- [ ] 路径测试脚本 `test_paths.py` 运行正常

## 更新日志

- **2024-12-19**: 修复模型保存路径，确保模型保存在项目根目录的 `models/` 文件夹
- **2024-12-19**: 统一训练和预测模块的路径配置
- **2024-12-19**: 添加路径测试脚本和文档说明 