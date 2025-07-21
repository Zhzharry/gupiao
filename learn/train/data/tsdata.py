import os
import pandas as pd
from collections import defaultdict
import glob

# ====== 配置部分 ======
# 数据源文件夹，存放每日行情的CSV文件
SOURCE_DIR = os.path.join(os.path.dirname(__file__), 'learn_csv')

# 输出文件夹，用于存放按股票代码组织的CSV文件
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'new')

# =====================

def process_daily_files():
    """
    读取SOURCE_DIR下的所有日线CSV文件，并按股票代码重新组织数据，
    然后将每只股票的数据保存到OUTPUT_DIR下的独立CSV文件中。
    """
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 使用字典来存储每只股票的数据，键为股票代码，值为DataFrame的行列表
    stock_data_map = defaultdict(list)
    
    # 获取所有日线CSV文件
    daily_files = glob.glob(os.path.join(SOURCE_DIR, '*_daily.csv'))
    
    if not daily_files:
        print(f"错误：在目录 '{SOURCE_DIR}' 中未找到 *_daily.csv 文件。")
        return

    print(f"找到 {len(daily_files)} 个日线文件，开始处理...")

    # 遍历每个日线文件
    for i, file_path in enumerate(sorted(daily_files)):
        print(f"正在处理第 {i+1}/{len(daily_files)} 个文件: {os.path.basename(file_path)}")
        try:
            # 读取日线文件
            daily_df = pd.read_csv(file_path)
            
            # 检查必要的列是否存在
            required_columns = ['tradingday', 'secucode', 'preclose', 'open', 'high', 'low', 'close', 'vol', 'amount', 'deals']
            if not all(col in daily_df.columns for col in required_columns):
                print(f"警告: 文件 {os.path.basename(file_path)} 缺少必要的列，已跳过。")
                continue

            # 遍历文件中的每一行
            for _, row in daily_df.iterrows():
                stock_code = row['secucode']
                # 将股票代码格式化为6位字符串，不足补零
                formatted_code = f"{int(stock_code):06d}"
                stock_data_map[formatted_code].append(row)

        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")

    print("\n所有文件读取完毕，开始写入新的CSV文件...")

    # 将每只股票的数据写入独立的CSV文件
    total_stocks = len(stock_data_map)
    for i, (code, rows) in enumerate(stock_data_map.items()):
        print(f"正在写入第 {i+1}/{total_stocks} 只股票: {code}")
        # 将行列表转换为DataFrame
        stock_df = pd.DataFrame(rows)
        # 按交易日升序排序
        stock_df = stock_df.sort_values(by='tradingday').reset_index(drop=True)
        
        # 定义输出路径
        output_path = os.path.join(OUTPUT_DIR, f"{code}.csv")
        
        # 保存为CSV
        stock_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n处理完成！共处理了 {total_stocks} 只股票的数据。")
    print(f"文件已保存至: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    process_daily_files() 