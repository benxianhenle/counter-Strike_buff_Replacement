import pandas as pd
import json


def extract_cs_data(excel_path):
    # 读取所有sheet，不解析表头
    sheets = pd.read_excel(excel_path, sheet_name=None, header=None)
    result = {}

    for sheet_name, df in sheets.items():
        sheet_data = []
        # 从第二行开始遍历（跳过表头）
        for _, row in df.iloc[1:].iterrows():
            # 提取皮肤名称（第二列）
            skin_name = row.iloc[1]

            # 设置这里控制普通皮肤[2:7]和暗金皮肤[8:12]
            numbers = row.iloc[8:12].dropna()
            if not numbers.empty:
                sheet_data.append({
                    "skin_name": skin_name if pd.notna(skin_name) else "",
                    "number": numbers.iloc[0]
                })

        result[sheet_name] = sheet_data

    # 生成JSON文件
    output_path = excel_path.replace(".xlsx", ".json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return output_path


# 执行提取操作
if __name__ == "__main__":
    excel_file = r"C:\Users\86150\Desktop\cs饰品\cs饰品编号\cs饰品编号.xlsx"
    output = extract_cs_data(excel_file)
    print(f"数据已成功导出至：{output}")