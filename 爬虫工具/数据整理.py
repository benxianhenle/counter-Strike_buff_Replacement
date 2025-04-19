import pandas as pd
import os
from openpyxl import load_workbook
import re

def process_weapon_box(file_path):
    # 读取原始数据
    df = pd.read_excel(file_path, sheet_name=0)

    # 预处理
    # 1. 将"普通"行的StatTrak设为0
    df.loc[df['磨损'] == '普通', '是否StatTrak™'] = 0
    # 2. 替换磨损值
    # 3. 去除皮肤名称中的括号及其内容
    df['磨损'] = df['磨损'].replace({'普通': '崭新出厂', 'StatTrak™': '崭新出厂'})
    df['皮肤名称'] = df['皮肤名称'].apply(remove_brackets)

    # 转换为目标格式
    # 创建目标数据结构
    target_columns = ['品质', '磨损', '崭新出厂', '略有磨损', '久经沙场', '破损不堪', '战痕累累',
                      'StatTrak™崭新出厂', 'StatTrak™略有磨损', 'StatTrak™久经沙场',
                      'StatTrak™破损不堪', 'StatTrak™战痕累累']

    # 按武器名称和品质分组
    grouped = df.groupby(['皮肤名称', '品质'])

    result = []
    for (weapon, quality), group in grouped:
        row = {
            '品质': quality,
            '磨损': weapon,
            '崭新出厂': '',
            '略有磨损': '',
            '久经沙场': '',
            '破损不堪': '',
            '战痕累累': '',
            'StatTrak™崭新出厂': '',
            'StatTrak™略有磨损': '',
            'StatTrak™久经沙场': '',
            'StatTrak™破损不堪': '',
            'StatTrak™战痕累累': ''
        }

        # 填充编号
        for _, record in group.iterrows():
            wear = record['磨损']
            stattrak = record['是否StatTrak™']
            col_prefix = 'StatTrak™' if stattrak else ''

            if wear in ['崭新出厂', '略有磨损', '久经沙场', '破损不堪', '战痕累累']:
                col_name = f"{col_prefix}{wear}"
                if pd.isna(row[col_name]) or row[col_name] == '':
                    row[col_name] = str(int(record['编号']))  # 初始化为单个编号
                else:
                    row[col_name] += f",{int(record['编号'])}"  # 修复：补全右括号

        result.append(row)  # 将填充完成的行添加到结果列表

    return pd.DataFrame(result, columns=target_columns)

def remove_brackets(text):
    # 使用正则表达式去除括号及其内容
    return re.sub(r'\([^)]*\)', '', text).strip()

def main():
    input_folder = r"C:\Users\86150\Desktop\cs饰品\wuqixiang"
    output_file = r"C:\Users\86150\Desktop\cs饰品\cs饰品编号\cs饰品编号.xlsx"

    # 定义品质顺序映射
    quality_order = {
        '违禁': 1,
        '隐秘': 2,
        '保密': 3,
        '受限': 4,
        '军规级': 5,
        '工业级': 6,
        '消费级': 7,
    }

    # 检查输出文件是否存在且是否为有效的 Excel 文件
    if os.path.exists(output_file):
        try:
            book = load_workbook(output_file)
            writer = pd.ExcelWriter(output_file, engine='openpyxl')
            writer.book = book
            writer.sheets = {ws.title: ws for ws in book.worksheets}
        except Exception as e:
            print(f"无法加载现有文件 {output_file}，将重新创建文件。错误信息: {e}")
            os.remove(output_file)  # 删除无效文件
            writer = pd.ExcelWriter(output_file, engine='openpyxl')
    else:
        writer = pd.ExcelWriter(output_file, engine='openpyxl')

    # 处理所有xlsx文件
    for file in os.listdir(input_folder):
        if file.endswith(".xlsx"):
            file_path = os.path.join(input_folder, file)
            df_processed = process_weapon_box(file_path)

            # 使用文件名作为sheet名（不带扩展名）
            sheet_name = os.path.splitext(file)[0]

            # 按品质顺序排序
            df_processed['品质顺序'] = df_processed['品质'].map(quality_order)
            df_processed = df_processed.sort_values(by='品质顺序').drop(columns=['品质顺序'])

            df_processed.to_excel(writer, sheet_name=sheet_name, index=False)

    # 保存结果
    writer.close()

if __name__ == "__main__":
    main()
