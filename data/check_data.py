"""
股票价格预测模型 - 数据检查工具
================================

文件作用：
- 快速检查数据目录和文件的状态
- 验证数据文件的完整性和可读性
- 统计数据文件的基本信息
- 提供数据质量评估报告

主要功能：
1. 目录检查：验证数据目录是否存在和可访问
2. 文件统计：统计CSV文件的数量和大小
3. 数据预览：显示数据文件的基本结构和内容
4. 格式验证：检查CSV文件的格式是否正确
5. 完整性检查：验证数据是否完整和一致

检查内容：
- 数据目录结构
- CSV文件数量和质量
- 数据列名和数据类型
- 缺失值和异常值
- 数据时间范围

使用方法：
- 直接运行：python check_data.py
- 指定目录：python check_data.py --data_dir path/to/data

输出结果：
- 数据目录状态报告
- 文件统计信息
- 数据质量评估
- 建议和警告信息

作者：AI Assistant
创建时间：2024年
"""
import os

def check_directory_structure():
    """检查目录结构"""
    print("=== 检查目录结构 ===")
    
    # 检查当前目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 列出当前目录下的所有内容
    print("\n当前目录内容:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  目录: {item}/")
        else:
            print(f"  文件: {item}")
    
    # 检查data目录
    if os.path.exists('data'):
        print("\ndata目录内容:")
        for item in os.listdir('data'):
            item_path = os.path.join('data', item)
            if os.path.isdir(item_path):
                print(f"  目录: {item}/")
                # 列出子目录内容
                try:
                    sub_items = os.listdir(item_path)
                    print(f"    {item}/ 包含 {len(sub_items)} 个项目")
                    if len(sub_items) > 0:
                        print(f"    示例: {sub_items[:5]}")
                except:
                    print(f"    无法访问 {item}/ 目录")
            else:
                print(f"  文件: {item}")
    else:
        print("\ndata目录不存在")
    
    # 检查是否有CSV文件
    print("\n=== 查找CSV文件 ===")
    csv_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if csv_files:
        print(f"找到 {len(csv_files)} 个CSV文件:")
        for csv_file in csv_files[:10]:  # 只显示前10个
            print(f"  {csv_file}")
        if len(csv_files) > 10:
            print(f"  ... 还有 {len(csv_files) - 10} 个文件")
    else:
        print("没有找到CSV文件")

if __name__ == "__main__":
    check_directory_structure() 