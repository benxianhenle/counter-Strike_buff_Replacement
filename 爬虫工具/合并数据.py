import pandas as pd
import os

# 定义文件夹路径
folder_path = r'C:\Users\86150\Desktop\cs饰品\wuqixiang'

# 获取文件夹中所有Excel文件的文件名
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]

# 创建一个空的DataFrame来存储合并后的数据
merged_data = pd.DataFrame()
# 定义磨损优先级
wear_priority = ['崭新出厂', '略有磨损', '久经沙场', '破损不堪', '战痕累累']
def adjust_wear(df):
    # 按 "皮肤名称" 和 "是否StatTrak™" 进行分组
    for (skin, stattrak), group in df.groupby(['皮肤名称', '是否StatTrak™']):
        # 找到该组中已有的合法磨损等级（去掉 "普通" 和 "StatTrak™"）
        existing_wear = [w for w in group['磨损'] if w in wear_priority]

        if not existing_wear:
            continue  # 如果没有找到有效的磨损数据，跳过

        # 找到该组中最低的磨损等级
        min_wear = min(existing_wear, key=lambda x: wear_priority.index(x))

        # 计算上一级磨损（索引减一）
        new_wear_index = wear_priority.index(min_wear) - 1
        if new_wear_index >= 0:
            new_wear = wear_priority[new_wear_index]
        else:
            new_wear = wear_priority[0]  # 最小值就是 "崭新出厂"

        # **修改 "普通" 仅适用于 `是否StatTrak™ == 0`**
        df.loc[(df['皮肤名称'] == skin) & (df['是否StatTrak™'] == stattrak) &
               (df['磨损'] == '普通') & (stattrak == 0), '磨损'] = new_wear

        # **修改 "StatTrak™" 仅适用于 `是否StatTrak™ == 1`**
        df.loc[(df['皮肤名称'] == skin) & (df['是否StatTrak™'] == stattrak) &
               (df['磨损'] == 'StatTrak™') & (stattrak == 1), '磨损'] = new_wear

    return df



# 遍历每个Excel文件并读取数据
for file in excel_files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_excel(file_path)
    # 去掉文件名的后缀
    file_name_without_ext = os.path.splitext(file)[0]
    # 添加一列，填充数据所在的文件名（去掉后缀）
    data['weapon_box'] = file_name_without_ext
    #将 皮肤名称 这一列的括号及其中的内容去除
    data['皮肤名称'] = data['皮肤名称'].str.replace(r'\(.*\)', '', regex=True)

    data.loc[data['磨损'] == '普通', '是否StatTrak™'] = 0
    #用adjust_wear函数将磨损进行修改
    data = adjust_wear(data)

    merged_data = pd.concat([merged_data, data], ignore_index=True)

# 将合并后的数据保存到一个新的Excel文件中
output_file_path = os.path.join(folder_path, r'C:\Users\86150\Desktop\cs饰品\cs饰品编号\对应编号.xlsx')
merged_data.to_excel(output_file_path, index=False)

print(f"合并完成，结果已保存到 {output_file_path}")
