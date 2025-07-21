import os
import pandas as pd
from tqdm import tqdm
import glob

def split_data_by_date():
    """
    读取 new/ 文件夹中的每个股票CSV文件，
    并根据指定的日期范围将其分割为学习、测试和评价三个部分，
    然后存入各自独立的文件夹中。
    """
    # --- 配置路径 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, 'new')
    
    # 定义三个目标输出目录
    learn_dir = os.path.join(script_dir, 'learn_csv2')
    test_dir = os.path.join(script_dir, 'test_csv2')
    adjustment_dir = os.path.join(script_dir, 'Adjustment_csv2')

    # 创建目标目录
    os.makedirs(learn_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(adjustment_dir, exist_ok=True)

    print(f"数据源目录: {os.path.abspath(source_dir)}")
    print(f"学习数据将保存至: {os.path.abspath(learn_dir)}")
    print(f"测试数据将保存至: {os.path.abspath(test_dir)}")
    print(f"评价数据将保存至: {os.path.abspath(adjustment_dir)}")

    # --- 定义日期范围 ---
    # pd.to_datetime 可以正确处理 YYYYMMDD 格式的整数
    learn_range = ('20230403', '20241231')
    test_range = ('20250102', '20250530')
    adjustment_range = ('20250603', '20250701')

    # --- 数据处理 ---
    source_files = glob.glob(os.path.join(source_dir, '*.csv'))
    if not source_files:
        print(f"错误: 在目录 '{os.path.abspath(source_dir)}' 中未找到任何 .csv 文件。")
        return

    print(f"\n找到 {len(source_files)} 个股票文件，开始分割...")

    for file_path in tqdm(sorted(source_files), desc="分割数据文件"):
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            # 将 'tradingday' 列转换为 datetime 对象以便于比较
            df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y%m%d')

            # 1. 分割学习数据
            learn_df = df[(df['tradingday'] >= learn_range[0]) & (df['tradingday'] <= learn_range[1])]
            if not learn_df.empty:
                # 转换回 YYYYMMDD 整数格式保存
                learn_df_save = learn_df.copy()
                learn_df_save['tradingday'] = learn_df_save['tradingday'].dt.strftime('%Y%m%d').astype(int)
                learn_df_save.to_csv(os.path.join(learn_dir, file_name), index=False, encoding='utf-8-sig')

            # 2. 分割测试数据
            test_df = df[(df['tradingday'] >= test_range[0]) & (df['tradingday'] <= test_range[1])]
            if not test_df.empty:
                test_df_save = test_df.copy()
                test_df_save['tradingday'] = test_df_save['tradingday'].dt.strftime('%Y%m%d').astype(int)
                test_df_save.to_csv(os.path.join(test_dir, file_name), index=False, encoding='utf-8-sig')

            # 3. 分割评价数据
            adjustment_df = df[(df['tradingday'] >= adjustment_range[0]) & (df['tradingday'] <= adjustment_range[1])]
            if not adjustment_df.empty:
                adj_df_save = adjustment_df.copy()
                adj_df_save['tradingday'] = adj_df_save['tradingday'].dt.strftime('%Y%m%d').astype(int)
                adj_df_save.to_csv(os.path.join(adjustment_dir, file_name), index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"\n处理文件 {file_name} 时出错: {e}")
            
    print(f"\n数据分割完成！")

if __name__ == "__main__":
    split_data_by_date()