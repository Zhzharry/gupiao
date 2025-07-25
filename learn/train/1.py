import os
import pandas as pd

base_dir = r"d:/programming/Workspace/gupiao/learn/train/data"
data_dir = os.path.join(base_dir, "data")
learn_dir = os.path.join(base_dir, "learn_csv3")
test_dir = os.path.join(base_dir, "test_csv3")
adjust_dir = os.path.join(base_dir, "Adjustment_csv3")

# 创建目标文件夹
os.makedirs(learn_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(adjust_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.lower().endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(file_path, dtype=str)
            if 'Date' not in df.columns:
                print(f"{filename} 缺少Date列，跳过")
                continue
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # 分割
            learn_df = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2022-06-30')]
            test_df = df[(df['Date'] >= '2022-07-01') & (df['Date'] <= '2023-06-30')]
            adjust_df = df[df['Date'] >= '2023-07-01']
            # 保存
            if not learn_df.empty:
                learn_df.to_csv(os.path.join(learn_dir, filename), index=False, encoding='utf-8-sig')
            if not test_df.empty:
                test_df.to_csv(os.path.join(test_dir, filename), index=False, encoding='utf-8-sig')
            if not adjust_df.empty:
                adjust_df.to_csv(os.path.join(adjust_dir, filename), index=False, encoding='utf-8-sig')
            print(f"{filename} 已分割")
        except Exception as e:
            print(f"{filename} 处理出错: {e}")

print("全部处理完成。")