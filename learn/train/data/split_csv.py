import os
import pandas as pd
from tqdm import tqdm
import glob

def split_csv_files():
    """将Adjustment_csv3中的CSV文件按列拆分为两组"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, 'Adjustment_csv3')
    first_dir = os.path.join(script_dir, 'Adjustment_csv3', 'first')
    second_dir = os.path.join(script_dir, 'Adjustment_csv3', 'second')
    
    # 创建目标文件夹
    os.makedirs(first_dir, exist_ok=True)
    os.makedirs(second_dir, exist_ok=True)
    
    # 定义两组列名
    price_columns = ['Date', 'secucode', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'change_ratio']
    news_columns = ['Date', 'secucode', 'Newsnum_Title_news1', 'Newsnum_Cont_news1', 'Posnews_All_news1', 
                   'Neunews_All_news1', 'Negnews_All_news1', 'Posnews_Ori_news1', 'Neunews_Ori_news1', 
                   'Negnews_Ori_news1', 'Newsnum_Title_news2', 'Newsnum_Cont_news2', 'Posnews_All_news2', 
                   'Neunews_All_news2', 'Negnews_All_news2', 'Posnews_Ori_news2', 'Neunews_Ori_news2', 
                   'Negnews_Ori_news2']
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(source_dir, '*.csv'))
    
    if not csv_files:
        print(f"错误: 在目录 '{os.path.abspath(source_dir)}' 中未找到任何 .csv 文件。")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件，开始拆分...")
    
    for file_path in tqdm(csv_files, desc="拆分CSV文件"):
        file_name = os.path.basename(file_path)
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查列是否存在
            available_price_cols = [col for col in price_columns if col in df.columns]
            available_news_cols = [col for col in news_columns if col in df.columns]
            
            if not available_price_cols:
                print(f"警告: 文件 {file_name} 中没有找到价格相关列")
                continue
                
            if not available_news_cols:
                print(f"警告: 文件 {file_name} 中没有找到新闻相关列")
                continue
            
            # 创建第一组数据（价格相关）
            df_price = df[available_price_cols].copy()
            price_file_path = os.path.join(first_dir, file_name)
            df_price.to_csv(price_file_path, index=False, encoding='utf-8-sig')
            
            # 创建第二组数据（新闻相关）
            df_news = df[available_news_cols].copy()
            news_file_path = os.path.join(second_dir, file_name)
            df_news.to_csv(news_file_path, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            print(f"\n处理文件 {file_name} 时出错: {e}")
    
    print(f"\n拆分完成！")
    print(f"价格相关数据保存到: {os.path.abspath(first_dir)}")
    print(f"新闻相关数据保存到: {os.path.abspath(second_dir)}")

if __name__ == '__main__':
    split_csv_files() 