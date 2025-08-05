import os
import pandas as pd
import glob

def check_data_dates():
    """检查测试数据的时间范围"""
    test_data_path = "data/test_csv3"
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(test_data_path, "*.csv"))
    print(f"找到 {len(csv_files)} 个测试数据文件")
    
    # 检查前几个文件的时间范围
    for i, csv_file in enumerate(csv_files[:5]):
        stock_code = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\n检查股票 {stock_code}:")
        
        try:
            df = pd.read_csv(csv_file)
            
            # 检查日期列
            if 'Date' in df.columns:
                date_col = 'Date'
            elif '日期' in df.columns:
                date_col = '日期'
            else:
                print(f"  错误：未找到日期列")
                continue
            
            # 转换日期格式
            df[date_col] = pd.to_datetime(df[date_col])
            
            print(f"  数据行数: {len(df)}")
            print(f"  日期范围: {df[date_col].min()} 到 {df[date_col].max()}")
            print(f"  最早日期: {df[date_col].min()}")
            print(f"  最晚日期: {df[date_col].max()}")
            
            # 检查是否包含2023年的数据
            df_2023 = df[df[date_col].dt.year == 2023]
            print(f"  2023年数据行数: {len(df_2023)}")
            if len(df_2023) > 0:
                print(f"  2023年日期范围: {df_2023[date_col].min()} 到 {df_2023[date_col].max()}")
            
        except Exception as e:
            print(f"  处理文件时出错: {e}")

if __name__ == "__main__":
    check_data_dates() 