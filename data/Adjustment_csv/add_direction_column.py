import os
import pandas as pd

# 目标目录
csv_dir = os.path.dirname(__file__)

for fname in os.listdir(csv_dir):
    if fname.endswith('.csv'):
        fpath = os.path.join(csv_dir, fname)
        df = pd.read_csv(fpath)
        if 'direction' not in df.columns:
            df['direction'] = (df['close'] > df['open']).astype(int)
            df.to_csv(fpath, index=False)
        else:
            print(f"{fname} 已有 direction 列，跳过。")
print("所有csv文件已处理完毕！") 