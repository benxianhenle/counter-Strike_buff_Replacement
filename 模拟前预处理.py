import pandas as pd
import 价格处理
import os
import glob
import datetime
from datetime import datetime
from collections import defaultdict
wear_mapping = {
    '崭新出厂': 1,
    '略有磨损': 2,
    '久经沙场': 3,
    '破损不堪': 4,
    '战痕累累': 5
}
def get_file_names(history_dir):
    # 获取所有文件名
    file_names = os.listdir(history_dir)
    return file_names

def fell_sunbathing( fell_df, input_ls):
    today = datetime.today().strftime('%Y-%m-%d')
    fell_df_ls = [f.split('_') for f in fell_df]
    # 过滤并排序
    filtered_files = []
    for parts in fell_df_ls :
        date_str = parts[0]
        #today ='2025-04-25'
        if date_str == today:
            continue  # 跳过今天的文件
        if len(parts)==0:
            continue
        if input_ls[4] not in parts:
            continue
        if input_ls[0] in parts:
            filtered_files.append(parts)

    # 按日期降序排序
    filtered_files.sort(key=lambda x: x[0], reverse=True)

    # 合并为文件名字符串（用下划线连接）
    result = ['_'.join(parts) for parts in filtered_files]
    return result


def read_excel_files(history_dir, fell_df_ls):
    """
    按照指定顺序读取Excel文件并返回DataFrame列表

    参数：
    history_dir: 文件路径
    fell_df_ls: 需要读取的Excel文件名列表

    返回：
    List[pd.DataFrame] 按输入顺序排列的DataFrame列表
    """
    df_list = []
    for filename in fell_df_ls:
        # 拼接完整文件路径
        file_path = os.path.join(history_dir, filename)
        # 读取Excel文件
        df = pd.read_excel(file_path)
        df =价格处理.main(df)
        df['skin_name'] = df['skin_name'].astype(str).str.strip()
        df['磨损'] = df['磨损'].astype(str).str.strip()
        df_list.append(df)
    return df_list


def supplement_missing_intervals(df, input_ls, history_dir='history_data'):
    reverse_wear_mapping = {v: k for k, v in wear_mapping.items()}
    all_ids = {1, 2, 3, 4, 5}
    supplemented_rows = []

    # 第一步：扫描历史文件
    fell_df = get_file_names(history_dir)
    fell_df_ls = fell_sunbathing(fell_df, input_ls)
    history_dfs = read_excel_files(history_dir, fell_df_ls)

    print(f"[信息] 成功加载 {len(history_dfs)} 个历史文件用于补充")

    # 第二步：建立 lookup
    lookup_names = df.set_index('skin_name_id')['skin_name'].to_dict()

    # 第三步：处理每个需要补充的皮肤
    for skin_id, group in df.groupby('skin_name_id'):
        missing_ids = all_ids - set(group['itemfloat_Interval_id'].astype(int))

        for missing_id in missing_ids:
            try:
                wear_name = reverse_wear_mapping[missing_id]
                skin_name = lookup_names[skin_id]

                # 在历史数据中查找
                best_record = None
                for hist_df in history_dfs:
                    match = hist_df[
                        (hist_df['skin_name'] == skin_name) &
                        (hist_df['磨损'] == wear_name)
                        ]
                    if not match.empty:
                        best_record = match.iloc[0]
                        break  # 找到就用最新的

                if best_record is not None:
                    new_row = {
                        'skin_name_id': skin_id,
                        'itemfloat_Interval_id': missing_id,
                        'price': best_record['price'],
                        'weapon_box_id': best_record['weapon_box_id'],
                        'quality': best_record['quality'],
                        'skin_name': skin_name
                    }
                    supplemented_rows.append(new_row)
                else:
                    print(f"⚠️ 未从历史数据中找到记录：skin_name={skin_name}，磨损={wear_name}")

            except Exception as e:
                print(f"❌ 处理异常 skin_id={skin_id}: {str(e)}")
                continue

    # 第四步：合并补充的数据
    if supplemented_rows:
        supplement_df = pd.DataFrame(supplemented_rows)
        df = pd.concat([df, supplement_df]).drop_duplicates(
            subset=['skin_name_id', 'itemfloat_Interval_id'],
            keep='last'
        )
        print(f"[完成] 成功补充了 {len(supplemented_rows)} 条缺失记录")

    return df




def clean_box(box_fill, itemfloat_fill,history_dir,input_ls):
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

    df['weapon_box'] = (
        df['weapon_box']
        .str.replace('[“”]', '', regex=True)
        .str.strip()
    )


    # 读取 itemfloat_fill 数据
    if not isinstance(itemfloat_fill, pd.DataFrame):
        itemfloat_fill = pd.read_excel(itemfloat_fill)

    # 标准化 skin_name 和 皮肤名称和 磨损范围
    df['skin_name'] = df['skin_name'].astype(str).str.strip()
    itemfloat_fill['皮肤名称'] = itemfloat_fill['皮肤名称'].astype(str).str.strip()
    df['磨损'] = df['磨损'].astype(str).str.strip()
    # 给 weapon_box 和 skin_name 以及磨损范围添加 id

    weapon_box_map = {name: idx for idx, name in enumerate(df['weapon_box'].unique())}
    df['weapon_box_id'] = df['weapon_box'].map(weapon_box_map)

    skin_name_map = {name: idx for idx, name in enumerate(df['skin_name'].unique())}
    df['skin_name_id'] = df['skin_name'].map(skin_name_map)

    df['itemfloat_Interval_id'] = df['磨损'].map(wear_mapping)
    # 找出最高优先级的 quality
    mapped_series = df['quality'].map(quality_order)
    valid = mapped_series.dropna()

    if valid.empty:# 没有任何值能映射到 quality_order，报错并打印原始 quality 值
        print("你可能需要更新你的cookie")
        raise ValueError(
                    f"[clean_box] 无法识别的 quality 值：{df['quality'].unique().tolist()}")
    idx = valid.idxmax()
    quality = df.loc[idx, 'quality']

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
    df_non_confidential = supplement_missing_intervals(df_non_confidential,input_ls,history_dir,)
    return df_non_confidential_ll, df_non_confidential, df_confidential_min_price
if __name__ == '__main__':
    box_fill = r"C:\Users\86150\Desktop\cs饰品\数据\2025-04-25_隐秘_崭新出厂_sgo_items2.xlsx"
    box_fill = 价格处理.main(box_fill)
    itemfloat_fill ="data/饰品磨损区间.xlsx"
    history_dir = r"C:\Users\86150\Desktop\cs饰品\数据"
    input_ls = ['隐秘', '保密', "崭新出厂"]
    clean_box(box_fill,itemfloat_fill, history_dir,input_ls)