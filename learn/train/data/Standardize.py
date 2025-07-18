import os
import pandas as pd
from pathlib import Path
import glob
from typing import List, Set, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_csv_files(directory: str) -> List[str]:
    """
    递归遍历目录，找到所有CSV文件
    
    Args:
        directory (str): 要搜索的目录路径
        
    Returns:
        List[str]: CSV文件路径列表
    """
    csv_files = []
    
    # 使用glob递归搜索所有CSV文件
    pattern = os.path.join(directory, '**', '*.csv')
    csv_files = glob.glob(pattern, recursive=True)
    
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    return csv_files

def load_csv_with_error_handling(file_path: str) -> pd.DataFrame:
    """
    安全地加载CSV文件，处理可能的错误
    
    Args:
        file_path (str): CSV文件路径
        
    Returns:
        pd.DataFrame: 加载的数据框，如果失败则返回空数据框
    """
    try:
        # 尝试不同的编码方式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"成功加载文件: {file_path} (编码: {encoding})")
                return df
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，记录错误
        logger.error(f"无法读取文件: {file_path}")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"加载文件 {file_path} 时出错: {str(e)}")
        return pd.DataFrame()

def get_stock_codes_from_file(file_path: str) -> Set[str]:
    """
    从单个CSV文件中获取所有股票代码
    
    Args:
        file_path (str): CSV文件路径
        
    Returns:
        Set[str]: 股票代码集合
    """
    df = load_csv_with_error_handling(file_path)
    
    if df.empty:
        return set()
    
    # 确保有secucode列
    if 'secucode' not in df.columns:
        logger.warning(f"文件 {file_path} 中没有找到 'secucode' 列")
        return set()
    
    # 返回股票代码集合，转换为字符串以确保一致性
    return set(df['secucode'].astype(str).unique())

def find_common_stocks(csv_files: List[str]) -> Set[str]:
    """
    找到在所有CSV文件中都存在的股票代码
    
    Args:
        csv_files (List[str]): CSV文件路径列表
        
    Returns:
        Set[str]: 在所有文件中都存在的股票代码集合
    """
    if not csv_files:
        logger.warning("没有找到CSV文件")
        return set()
    
    # 获取第一个文件的股票代码作为初始集合
    common_stocks = get_stock_codes_from_file(csv_files[0])
    logger.info(f"第一个文件包含 {len(common_stocks)} 只股票")
    
    # 与其他文件的股票代码取交集
    for i, file_path in enumerate(csv_files[1:], 1):
        file_stocks = get_stock_codes_from_file(file_path)
        common_stocks = common_stocks.intersection(file_stocks)
        logger.info(f"处理第 {i+1} 个文件后，共同股票数量: {len(common_stocks)}")
        
        if len(common_stocks) == 0:
            logger.warning("没有找到在所有文件中都存在的股票")
            break
    
    logger.info(f"最终找到 {len(common_stocks)} 只在所有文件中都存在的股票")
    return common_stocks

def clean_single_file(file_path: str, common_stocks: Set[str], output_dir: str) -> bool:
    """
    清洗单个CSV文件，只保留共同股票的数据，直接覆盖原始文件
    
    Args:
        file_path (str): 输入文件路径
        common_stocks (Set[str]): 要保留的股票代码集合
        output_dir (str): （已废弃，不再使用）
        
    Returns:
        bool: 是否成功处理
    """
    df = load_csv_with_error_handling(file_path)
    
    if df.empty:
        return False
    
    # 确保有secucode列
    if 'secucode' not in df.columns:
        logger.warning(f"文件 {file_path} 中没有找到 'secucode' 列")
        return False
    
    # 转换股票代码为字符串以确保一致性
    df['secucode'] = df['secucode'].astype(str)
    
    # 过滤出共同股票的数据
    original_count = len(df)
    df_filtered = df[df['secucode'].isin(list(common_stocks))]
    filtered_count = len(df_filtered)
    
    logger.info(f"文件 {os.path.basename(file_path)}: 原始记录 {original_count}, 过滤后记录 {filtered_count}")
    
    # 直接覆盖原始文件
    try:
        df_filtered.to_csv(file_path, index=False, encoding='utf-8')
        logger.info(f"清洗后的数据已覆盖原文件: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存文件 {file_path} 时出错: {str(e)}")
        return False

def create_summary_report(csv_files: List[str], common_stocks: Set[str], output_dir: str):
    """
    创建数据清洗摘要报告
    
    Args:
        csv_files (List[str]): 处理的CSV文件列表
        common_stocks (Set[str]): 共同股票集合
        output_dir (str): 输出目录
    """
    summary = {
        '处理的文件数量': len(csv_files),
        '共同股票数量': len(common_stocks),
        '文件列表': [os.path.basename(f) for f in csv_files],
        '共同股票代码': sorted(list(common_stocks))
    }
    
    # 保存摘要报告
    summary_file = os.path.join(output_dir, 'cleaning_summary.txt')
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("股票数据清洗摘要报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理的文件数量: {summary['处理的文件数量']}\n")
            f.write(f"共同股票数量: {summary['共同股票数量']}\n\n")
            
            f.write("处理的文件列表:\n")
            for file_name in summary['文件列表']:
                f.write(f"- {file_name}\n")
            
            f.write(f"\n共同股票代码 ({len(common_stocks)} 只):\n")
            for i, code in enumerate(summary['共同股票代码'], 1):
                f.write(f"{i:4d}. {code}\n")
        
        logger.info(f"摘要报告已保存到: {summary_file}")
        
    except Exception as e:
        logger.error(f"保存摘要报告时出错: {str(e)}")

def main():
    """
    主函数：执行完整的数据清洗流程
    """
    # 配置参数
    current_directory = "."  # 当前目录，可以修改为其他目录
    output_directory = "cleaned_data"  # 输出目录
    
    logger.info("开始股票数据清洗流程")
    logger.info(f"搜索目录: {os.path.abspath(current_directory)}")
    
    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)
    
    # 1. 找到所有CSV文件
    csv_files = find_csv_files(current_directory)
    
    if not csv_files:
        logger.error("没有找到任何CSV文件")
        return
    
    # 2. 找到共同股票
    common_stocks = find_common_stocks(csv_files)
    
    if not common_stocks:
        logger.error("没有找到在所有文件中都存在的股票")
        return
    
    # 3. 清洗每个文件
    success_count = 0
    for file_path in csv_files:
        if clean_single_file(file_path, common_stocks, output_directory):
            success_count += 1
    
    # 4. 生成摘要报告
    create_summary_report(csv_files, common_stocks, output_directory)
    
    # 5. 输出最终结果
    logger.info("=" * 50)
    logger.info("数据清洗完成!")
    logger.info(f"处理文件数量: {len(csv_files)}")
    logger.info(f"成功处理: {success_count}")
    logger.info(f"共同股票数量: {len(common_stocks)}")
    logger.info(f"清洗后的文件保存在: {os.path.abspath(output_directory)}")
    
    # 打印一些共同股票代码作为示例
    if common_stocks:
        sample_stocks = sorted(list(common_stocks))[:10]  # 显示前10个
        logger.info("部分共同股票代码示例:")
        for code in sample_stocks:
            logger.info(f"  - {code}")
        if len(common_stocks) > 10:
            logger.info(f"  ... 还有 {len(common_stocks) - 10} 只股票")

if __name__ == "__main__":
    main()