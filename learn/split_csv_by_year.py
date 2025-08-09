import os
import pandas as pd
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def split_csv_by_year(source_directory, output_base_directory):
    """
    按照年份分割CSV文件并组织到不同文件夹
    
    Args:
        source_directory (str): 源数据目录
        output_base_directory (str): 输出基础目录
    """
    # 检查源目录是否存在
    if not os.path.exists(source_directory):
        print(f"错误：源目录 {source_directory} 不存在")
        return
    
    # 创建输出目录结构
    train_dir = os.path.join(output_base_directory, "train")
    val_dir = os.path.join(output_base_directory, "validation")
    test_dir = os.path.join(output_base_directory, "test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(source_directory) if f.endswith('.csv') and os.path.isfile(os.path.join(source_directory, f))]
    
    if not csv_files:
        print(f"目录 {source_directory} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    print("开始按年份分割文件...")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 定义年份范围
    train_years = list(range(2015, 2022))  # 2015-2021
    val_years = [2022]  # 2022
    test_years = [2023, 2024]  # 2023-2024
    
    for filename in csv_files:
        file_path = os.path.join(source_directory, filename)
        
        try:
            # 尝试读取CSV文件
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                except:
                    print(f"无法读取文件 {filename}，跳过")
                    error_count += 1
                    continue
            
            # 查找日期列
            date_column = None
            possible_date_names = ['Date', '日期', 'date', 'time', 'Time', '时间']
            
            for col in df.columns:
                if col in possible_date_names:
                    date_column = col
                    break
            
            if date_column is None:
                print(f"文件 {filename} 中没有找到日期列，跳过")
                skipped_count += 1
                continue
            
            # 转换日期格式
            try:
                df[date_column] = pd.to_datetime(df[date_column])
            except:
                print(f"文件 {filename} 日期格式转换失败，跳过")
                skipped_count += 1
                continue
            
            # 提取年份
            df['year'] = df[date_column].dt.year
            
            # 检查年份范围
            available_years = df['year'].unique()
            
            # 检查是否包含所需年份范围的数据
            has_train_data = any(year in available_years for year in train_years)
            has_val_data = any(year in available_years for year in val_years)
            has_test_data = any(year in available_years for year in test_years)
            
            # 如果没有任何所需年份的数据，跳过该文件
            if not (has_train_data or has_val_data or has_test_data):
                print(f"文件 {filename} 不包含2015-2024年的数据，删除文件")
                os.remove(file_path)
                skipped_count += 1
                continue
            
            # 分割数据
            train_data = df[df['year'].isin(train_years)]
            val_data = df[df['year'].isin(val_years)]
            test_data = df[df['year'].isin(test_years)]
            
            # 保存分割后的数据
            if not train_data.empty:
                train_file_path = os.path.join(train_dir, filename)
                train_data = train_data.drop('year', axis=1)  # 移除临时年份列
                train_data.to_csv(train_file_path, index=False, encoding='utf-8')
                print(f"已保存训练集: {filename} ({len(train_data)} 行)")
            
            if not val_data.empty:
                val_file_path = os.path.join(val_dir, filename)
                val_data = val_data.drop('year', axis=1)  # 移除临时年份列
                val_data.to_csv(val_file_path, index=False, encoding='utf-8')
                print(f"已保存验证集: {filename} ({len(val_data)} 行)")
            
            if not test_data.empty:
                test_file_path = os.path.join(test_dir, filename)
                test_data = test_data.drop('year', axis=1)  # 移除临时年份列
                test_data.to_csv(test_file_path, index=False, encoding='utf-8')
                print(f"已保存测试集: {filename} ({len(test_data)} 行)")
            
            processed_count += 1
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            error_count += 1
    
    print(f"\n分割完成！")
    print(f"成功处理: {processed_count} 个文件")
    print(f"跳过/删除: {skipped_count} 个文件")
    print(f"错误: {error_count} 个文件")
    
    # 统计各数据集的文件数量
    train_files = len([f for f in os.listdir(train_dir) if f.endswith('.csv')])
    val_files = len([f for f in os.listdir(val_dir) if f.endswith('.csv')])
    test_files = len([f for f in os.listdir(test_dir) if f.endswith('.csv')])
    
    print(f"\n数据集统计:")
    print(f"训练集 (2015-2021): {train_files} 个文件")
    print(f"验证集 (2022): {val_files} 个文件")
    print(f"测试集 (2023-2024): {test_files} 个文件")
    
    print(f"\n输出目录:")
    print(f"训练集: {train_dir}")
    print(f"验证集: {val_dir}")
    print(f"测试集: {test_dir}")

def main():
    # 源数据目录
    source_directory = r"D:\programming\Workspace\gupiao\learn\train\data2"
    
    # 输出基础目录
    output_base_directory = r"D:\programming\Workspace\gupiao\learn\train\split_data"
    
    print("CSV文件按年份分割工具")
    print("=" * 60)
    print(f"源数据目录: {source_directory}")
    print(f"输出基础目录: {output_base_directory}")
    print()
    print("分割规则:")
    print("  训练集: 2015-2021年")
    print("  验证集: 2022年")
    print("  测试集: 2023-2024年")
    print("  不包含2015-2024年数据的文件将被删除")
    print()
    
    # 直接执行，无需确认
    split_csv_by_year(source_directory, output_base_directory)

if __name__ == "__main__":
    main()
