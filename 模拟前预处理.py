import pandas as pd

def clean_box(box_fill, itemfloat_fill):
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

    # 读取 box_fill 数据
    if isinstance(box_fill, pd.DataFrame):
        df = box_fill.copy()
    else:
        df = pd.read_excel(box_fill)

    # 读取 itemfloat_fill 数据
    if not isinstance(itemfloat_fill, pd.DataFrame):
        itemfloat_fill = pd.read_excel(itemfloat_fill)

    # 标准化 skin_name 和 皮肤名称
    df['skin_name'] = df['skin_name'].astype(str).str.strip()
    itemfloat_fill['皮肤名称'] = itemfloat_fill['皮肤名称'].astype(str).str.strip()

    # 给 weapon_box 和 skin_name 编号
    weapon_box_map = {name: idx for idx, name in enumerate(df['weapon_box'].unique())}
    df['weapon_box_id'] = df['weapon_box'].map(weapon_box_map)

    skin_name_map = {name: idx for idx, name in enumerate(df['skin_name'].unique())}
    df['skin_name_id'] = df['skin_name'].map(skin_name_map)

    # 找出最高优先级的 quality
    mapped_series = df['quality'].map(quality_order)
    quality = df.loc[mapped_series.idxmax(), 'quality']

    # 删除缺失数据
    df = df.dropna()

    # 筛选出 quality 为目标值的数据，并按 weapon_box 保留最低价
    df_confidential = df[df['quality'] == quality]
    df_confidential_min_price = df_confidential.loc[
        df_confidential.groupby('weapon_box')['price'].idxmin()
    ]

    # 筛选非目标 quality 并去重
    df_non_confidential = df[df['quality'] != quality]
    df_non_confidential_ll = df_non_confidential.drop_duplicates(subset=['skin_name'])

    # 提取 itemfloat 表的关键列并重命名
    itemfloat_selected = itemfloat_fill[['皮肤名称', '最小磨损', '最大磨损']].rename(
        columns={
            '皮肤名称': 'skin_name',
            '最小磨损': 'min_itemfloa',
            '最大磨损': 'max_itemfloa'
        }
    )

    # 合并 itemfloat 数据进 df 和 df_confidential_min_price
    df_non_confidential_ll = pd.merge(df_non_confidential_ll, itemfloat_selected, on='skin_name', how='left')
    df_confidential_min_price = pd.merge(df_confidential_min_price, itemfloat_selected, on='skin_name', how='left')

    return df_non_confidential_ll, df_non_confidential, df_confidential_min_price
