
import pandas as pd
import re

# 文件路径配置
input_path = "C:/Users/86136/Desktop/wifi/写作/准确率对比表.xlsx"
output_path = "D:/Python programs/formatted_accuracy.csv"

# 读取Sheet1的所有数据（强制字符串类型）
df = pd.read_excel(
    input_path,
    sheet_name="Sheet1",
    header=None,
    dtype=str,
    engine='openpyxl'
)

formatted_data = []
current_algorithm = None
data_start_row = None

# 遍历每一行寻找表格起始标记
for idx, row in df.iterrows():
    # 检测算法表格标题行（列F包含"准确率对比表"）
    if "准确率对比表" in str(row[5]):
        # 提取算法名称（示例："未适应时准确率对比表" -> "未适应时"）
        current_algorithm = re.sub(r'准确率对比表.*', '', str(row[5])).strip()
        data_start_row = idx + 2  # 数据起始行（标题下2行）
        continue

    # 处理数据区域（每个表格3行数据）
    if current_algorithm and (data_start_row <= idx <= data_start_row + 2):
        # Source值应为1/2/3（根据行偏移计算）
        source = idx - data_start_row + 1

        # 提取三列数据（F-H列，对应索引5-7）
        target1 = row[6]  # Target1
        target2 = row[7]  # Target2
        target3 = row[8]  # Target3

        # 处理每个Target的数据
        for target_idx, value in enumerate([target1, target2, target3], start=1):
            # 跳过空值或无效格式
            if pd.isna(value) or "±" not in value:
                continue

            # 分割准确率和标准差
            accuracy, std = str(value).split(" ± ")
            formatted_data.append({
                "Algorithm": current_algorithm,
                "Source": source,
                "Target": f"Target{target_idx}",
                "Accuracy": float(accuracy.strip('%')),
                "Std": float(std.strip('%'))
            })

# 转换为DataFrame并保存
final_df = pd.DataFrame(formatted_data)
final_df.to_csv(output_path, index=False)

# 打印完整数据验证
pd.set_option('display.max_rows', None)
print("=== 完整数据 ===")
print(final_df.to_string(index=False))