import pandas as pd
import json
def main(file_path,skin_lst):
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

    def get_numbers_from_excel(file_path, text_1, text_2, quality,ls):
        # 读取所有工作表
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        selected_sheets = list(all_sheets.items())[ls[0]:ls[1]]

        numbers = []

        wear_levels = ["崭新出厂", "略有磨损", "久经沙场", "破损不堪", "战痕累累"]

        for sheet_name, df in selected_sheets:
            filtered1 = df[df['品质'] == text_1]
            filtered2 = df[df['品质'] == text_2]

            # 决定哪个品质更高级（数值更小）
            if quality_order[text_1] >= quality_order[text_2]:
                primary = filtered1
                secondary = filtered2
            else:
                primary = filtered2
                secondary = filtered1

            # 先处理 primary 中的指定磨损等级
            for _, row in primary.iterrows():
                if pd.notna(row.get(quality)):
                    numbers.append(int(row[quality]))

            # 再从 secondary 中所有磨损等级找非空的
            for _, row in secondary.iterrows():
                for wear in wear_levels:
                    if pd.notna(row.get(wear)):
                        numbers.append(int(row[wear]))

        return numbers

    # 使用示例

    result = get_numbers_from_excel(file_path, skin_lst[0], skin_lst[1], skin_lst[2],skin_lst[3])

    # 生成JSON并保存
    with open("csgoItemsInfoFixed1.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("处理完成，共找到 {} 个编号".format(len(result)))
    return len(result)
if __name__ == "__main__":
    file_path = r"C:\Users\86150\Desktop\cs饰品\cs饰品编号\cs饰品编号.xlsx"
    skin_lst = ['隐秘','保密',"崭新出厂"]
    main(file_path,skin_lst)
