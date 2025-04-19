import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
# 输入和输出文件路径
input_file = parser.parse_args().input
output_file = parser.parse_args().output

# 定义原表名称（新增炉渣1和炉渣2）
sheets = ['收益率', '成本', '出货概率', '磨损', '期望收益', '炉渣1', '炉渣2','价格']

# 读取所有工作表
dfs = {sheet: pd.read_excel(input_file, sheet_name=sheet) for sheet in sheets}

# 获取所有武器箱组合和皮肤名称的组合（基准）
prob_combinations = dfs['出货概率'][['武器箱组合', '皮肤名称']].drop_duplicates()
wear_combinations = dfs['磨损'][['武器箱组合', '皮肤名称']].drop_duplicates()
base_combinations = pd.concat([prob_combinations, wear_combinations]).drop_duplicates()

# 目标列名（确保顺序正确）
target_columns = ['0:10', '1:9', '2:8', '3:7', '4:6', '5:5', '6:4', '7:3', '8:2', '9:1', '10:0']

def itemfloat_if(itemfloat_num):
    if itemfloat_num<0.07:
        return "崭新出厂"
    elif itemfloat_num<0.15:
        return "略有磨损"
    elif itemfloat_num<0.38:
        return "久经沙场"
    elif itemfloat_num<0.45:
        return "破损不堪"
    else:
        return  "战痕累累"


# 预处理炉渣表的列名（避免与其他表列名冲突）
slag_tables = ['炉渣1', '炉渣2']
price = dfs['价格']
for table in slag_tables:
    # 提取炉渣表中非“武器箱组合”的列名，并添加前缀
    slag_columns = [col for col in dfs[table].columns if col != '武器箱组合']
    dfs[table] = dfs[table].rename(columns={col: f'{table}_{col}' for col in slag_columns})

# 创建Excel writer对象
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for col in target_columns:
        new_data = []
        # 遍历每个基准组合
        for _, row in base_combinations.iterrows():
            weapon = row['武器箱组合']
            skin = row['皮肤名称']

            # 从各表获取数据
            # 原表数据（同之前逻辑）
            # 出货概率
            prob_value = dfs['出货概率'][(dfs['出货概率']['武器箱组合'] == weapon) &
                                         (dfs['出货概率']['皮肤名称'] == skin)][col].values
            prob_value = prob_value[0] if len(prob_value) > 0 else None

            # 磨损
            wear_value = dfs['磨损'][(dfs['磨损']['武器箱组合'] == weapon) &
                                     (dfs['磨损']['皮肤名称'] == skin)][col].values
            wear_value = wear_value[0] if len(wear_value) > 0 else None

            # 其他表（无皮肤名称）
            # 收益率、成本、期望收益（简化为示例）
            yield_value = dfs['收益率'][dfs['收益率']['武器箱组合'] == weapon][col].values[0] if not dfs['收益率'][
                dfs['收益率']['武器箱组合'] == weapon].empty else None
            cost_value = dfs['成本'][dfs['成本']['武器箱组合'] == weapon][col].values[0] if not dfs['成本'][
                dfs['成本']['武器箱组合'] == weapon].empty else None
            exp_value = dfs['期望收益'][dfs['期望收益']['武器箱组合'] == weapon][col].values[0] if not dfs['期望收益'][
                dfs['期望收益']['武器箱组合'] == weapon].empty else None

            # 新增炉渣表数据（根据武器箱组合直接关联）
            slag_data = {}
            for table in slag_tables:
                # 获取炉渣表中对应武器箱组合的行
                slag_row = dfs[table][dfs[table]['武器箱组合'] == weapon]
                # 提取所有列数据（排除武器箱组合）
                for slag_col in slag_row.columns:
                    if slag_col != '武器箱组合':
                        slag_data[slag_col] = slag_row[slag_col].values[0] if not slag_row.empty else None

            # 在price的skin_name 中找寻skin所在的行
            price_row = price[price['skin_name'] == skin]
            # 判断磨损范围
            itemfloat112=itemfloat_if(wear_value)
            #在price_row中找寻itemfloat112列对应的值
            price_num = price_row[itemfloat112].values[0] if not price_row.empty else None
            # 构建新行
            new_row = {
                '武器箱组合': weapon,
                '组合出货皮肤': skin,
                '皮肤均值': price_num ,
                '组合收益率': yield_value,
                '组合成本': cost_value,
                '皮肤概率': prob_value,
                '磨损区间': itemfloat112,
                '磨损值': wear_value,
                '期望收益': exp_value,
                '收益':exp_value-cost_value,
                **slag_data  # 合并炉渣表数据
            }
            new_data.append(new_row)

        # 创建DataFrame并保存到Excel
        new_df = pd.DataFrame(new_data)
        sheet_name = col.replace(':', '_')
        new_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("处理完成！输出文件已保存为", output_file)