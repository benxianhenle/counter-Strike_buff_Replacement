import pandas as pd

# 读取数据

def main(file_path,skin_file="data/cs饰品编号/对应编号.xlsx"):
    # 读取物品价格数据
    if isinstance(file_path, pd.DataFrame):
        df_items = file_path.copy()
    else:
        df_items = pd.read_excel(file_path)
    # 提取所需列
    df_items = df_items[['goods_id', 'price', 'itemfloat']]

    # 计算相同 goods_id 的平均价格和磨损
    df_avg_price = df_items.groupby('goods_id', as_index=False).agg({'price': 'mean', 'itemfloat': 'mean'})
    #print(df_avg_price.head())
    # 读取皮肤数据
    df_skins = pd.read_excel(skin_file)
    # 统一列名
    df_skins.rename(columns={'编号': 'goods_id'}, inplace=True)
    # 提取编号、皮肤名称和品质
    df_skins = df_skins[['goods_id', '皮肤名称', '品质', 'weapon_box',"磨损"]]
    # 重命名列
    df_skins.rename(columns={'皮肤名称': 'skin_name', '品质': 'quality'}, inplace=True)

    # 合并数据
    df_merged = pd.merge(df_avg_price, df_skins, on='goods_id', how='left')
    # 调整列顺序
    df_merged = df_merged[['skin_name', 'weapon_box', 'quality', 'goods_id', 'price', 'itemfloat',"磨损"]]
    return df_merged

if __name__ == '__main__':
    skin_file = r"C:\Users\86150\Desktop\cs饰品\cs饰品编号\对应编号.xlsx"
    items_file = r"C:\Users\86150\Desktop\cs饰品\数据\2025-04-07_csgo_items2.xlsx"
    output_file = r"C:\Users\86150\Desktop\cs饰品\分析表\processed_csgo_items.xlsx"

    df=main(items_file, skin_file)
    print(df.head(10))
    # 保存到 Excel
    df.to_excel(output_file, index=False)

    print(f"处理完成，文件已保存为 {output_file}")


