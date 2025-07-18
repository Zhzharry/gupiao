import os
import pandas as pd

def find_zero_value_stocks(csv_path):
    """检测单个CSV文件中的零值股票"""
    zero_stocks = set()
    try:
        df = pd.read_csv(csv_path, dtype={'secucode': str})
        
        # 定义需要检查的列
        check_cols = ['tradingday', 'secucode', 'preclose', 'open', 'high', 
                      'low', 'close', 'vol', 'amount', 'deals']
        
        # 检查是否存在必须的列
        missing_cols = [col for col in check_cols if col not in df.columns]
        if missing_cols:
            print(f"警告: 文件 {os.path.basename(csv_path)} 缺少列 {missing_cols}，跳过处理")
            return zero_stocks
        
        # 找出包含零值的行
        zero_mask = (df[check_cols] == 0).any(axis=1)
        zero_rows = df[zero_mask]
        
        # 收集有问题的股票代码
        for secucode in pd.Series(zero_rows['secucode']).unique():
            # 确保股票代码有效（非空且非零值字符串）
            if secucode and secucode != '0' and secucode != 0:
                zero_stocks.add(str(secucode))
                
    except Exception as e:
        print(f"处理文件 {csv_path} 时出错: {str(e)}")
    
    return zero_stocks

def remove_stocks_from_all_files(directory, stocks_to_remove):
    """从所有CSV文件中删除指定的股票"""
    if not stocks_to_remove:
        print("没有需要删除的股票代码")
        return
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(csv_path, dtype={'secucode': str})
                    
                    # 记录原始行数
                    original_count = len(df)
                    
                    # 删除指定股票
                    df = df[~df['secucode'].isin(stocks_to_remove)]
                    
                    # 保存处理后的文件
                    df.to_csv(csv_path, index=False)
                    print(f"文件: {file} | 原始行数: {original_count} | 保留行数: {len(df)} | 删除行数: {original_count - len(df)}")
                    
                except Exception as e:
                    print(f"处理文件 {csv_path} 时出错: {str(e)}")

def main():
    base_dir = r"D:\programming\Workspace\gupiao\learn\train\data"
    all_zero_stocks = set()
    
    # 第一步: 收集所有有问题的股票代码
    print("开始扫描文件检测零值股票...")
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                zero_stocks = find_zero_value_stocks(csv_path)
                if zero_stocks:
                    print(f"在文件 {file} 中发现零值股票: {', '.join(zero_stocks)}")
                    all_zero_stocks.update(zero_stocks)
    
    # 输出检测结果
    if all_zero_stocks:
        print(f"\n发现 {len(all_zero_stocks)} 只有问题的股票: {', '.join(all_zero_stocks)}")
    else:
        print("\n未发现含有零值的股票")
        return
    
    # 第二步: 从所有文件中删除这些股票
    print("\n开始从所有文件中删除问题股票...")
    remove_stocks_from_all_files(base_dir, all_zero_stocks)
    print("\n处理完成! 所有问题股票已删除")

if __name__ == "__main__":
    main()