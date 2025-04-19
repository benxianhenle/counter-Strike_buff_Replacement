import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
# 读取原始Excel文件
file_path =  parser.parse_args().input
output_path = parser.parse_args().output
sheets = ['收益率', '成本', '出货概率', '磨损','炉渣1','炉渣2']

# 读取所有工作表数据
data = {}
sheet_js = None
for sheet in sheets:
    try:
        data[sheet] = pd.read_excel(file_path, sheet_name=sheet)
    except:
        pass  # 处理不存在的工作表

# 定义需要直接复制的附加表
additional_sheets = ['价格']  # 添加其他表名

# 批量读取附加表
additional_data = {}
for sheet in additional_sheets:
    try:
        additional_data[sheet] = pd.read_excel(file_path, sheet_name=sheet)
    except:
        print(f"警告：未找到表 {sheet}")
# 处理收益率表
yield_df = data.get('收益率')
if yield_df is None:
    raise ValueError("缺少'收益率'表")

# 筛选B-L列中存在大于1的行
mask = (yield_df.iloc[:, 1:13] > 1).any(axis=1)  # B-L列对应索引1-11
#打印前10行

filtered_rows = yield_df[mask]
filtered_names = filtered_rows.iloc[:, 0].tolist()  # A列值
#打印第1列

# 筛选其他表中A列匹配的行
filtered_data = {'收益率': filtered_rows}
for sheet in ['成本', '出货概率', '磨损','炉渣1','炉渣2']:
    df = data.get(sheet)
    if df is not None:
        filtered_df = df[df.iloc[:, 0].isin(filtered_names)]
        filtered_data[sheet] = filtered_df

# 计算期望收益表（需要成本表存在）
if '成本' in data and '收益率' in filtered_data:
    cost_df = data['成本']
    # 确保成本表也按相同条件筛选
    filtered_cost = cost_df[cost_df.iloc[:, 0].isin(filtered_names)]

    # 对齐索引并计算
    merged = filtered_rows.set_index(filtered_rows.columns[0])
    merged_cost = filtered_cost.set_index(filtered_cost.columns[0])
    expected = merged.iloc[:, 0:13] * merged_cost.iloc[:, 0:13]
    expected.reset_index(inplace=True)
    filtered_data['期望收益'] = expected

# 保存到新文件
with pd.ExcelWriter(output_path) as writer:
    for sheet_name, df in filtered_data.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    for sheet_name, df in additional_data.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"处理完成，结果已保存到 {output_path}")