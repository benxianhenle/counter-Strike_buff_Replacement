import pandas as pd
import numpy as np
import itertools
import logging
import 模拟前预处理
import 价格处理
import argparse
from datetime import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from concurrent.futures import ProcessPoolExecutor


######################################## 函数运算 ########################################################################
# 用来计算磨损
def get_wear_values_by_skin_name(max_itemfloa, min_itemfloa,itemfloat_avedge,Weight_ls):
    # Step 1: 权重乘积，输出 shape: (n,)
    offset_weights = (itemfloat_avedge @ Weight_ls.T).flatten()  # shape: (n,)
    # Step 2: 变形为 (n, 1, 1) 以便广播
    offset_weights = offset_weights[:, None, None]  # shape: (n, 1, 1)
    # Step 3: 扩展 max/min 为 (1, m, k)
    delta = (max_itemfloa - min_itemfloa)[None, :, :]  # shape: (1, m, k)
    min_itemfloa = min_itemfloa[None, :, :]            # shape: (1, m, k)
    # Step 4: 广播后运算 -> shape: (n, m, k)
    wear_values = delta * offset_weights + min_itemfloa
    return wear_values  # shape: (n, m, k）

def get_wear_label(itemfloat_num):
    if itemfloat_num < 0.07:
        return "崭新出厂"
    elif itemfloat_num < 0.15:
        return "略有磨损"
    elif itemfloat_num < 0.38:
        return "久经沙场"
    elif itemfloat_num < 0.45:
        return "破损不堪"
    else:
        return "战痕累累"

def get_thread_count(num_tasks):
    if num_tasks > 10000:
        return 10
    elif num_tasks > 5000:
        return 5
    elif num_tasks > 1000:
        return 2
    else:
        return 1
# 计算磨损矩阵
def convert_wear_to_price_fast(wear_matrix, skin_id, secret_df):
    wear_matrix = np.array(wear_matrix)
    n, m, k = wear_matrix.shape
    # 将 secret 表转为映射字典 {(skin_name_id, 磨损): price}
    price_map = {
        (row['skin_name_id'], row['磨损']): row['price']
        for _, row in secret_df.iterrows()
    }
    # 将 skin_id 列表转为二维 numpy 数组，shape = (m, k)
    skin_id_matrix = np.zeros((m, k), dtype=int)
    for j in range(m):
        for l in range(len(skin_id[j])):
            skin_id_matrix[j, l] = skin_id[j][l]
    # 预分配结果矩阵
    result = np.zeros_like(wear_matrix)
    # 向量化遍历所有位置
    for j in range(m):
        for l in range(k):
            skin = skin_id_matrix[j, l]
            if skin == 0:
                continue
            # 取出所有 n 个组合下第 j, l 位置的磨损值
            wear_column = wear_matrix[:, j, l]
            # 转换为磨损等级列表
            wear_labels = np.vectorize(get_wear_label)(wear_column)
            # 从映射中查找价格
            prices = np.array([
                price_map.get((skin, label), 0)
                for label in wear_labels
            ])
            # 填入结果
            result[:, j, l] = prices
    return result

#期望矩阵计算
def calc_expected_profit(price_matrix, Weight_ls):
    n, m, k = price_matrix.shape

    # Step 1: 平均每行的价格 -> shape: (n, m)
    avg_price_per_row = price_matrix.mean(axis=2)

    # Step 2: 每行乘以对应的权重 -> shape: (n, m)
    weighted_price = avg_price_per_row * Weight_ls

    # Step 3: 按行求和 -> shape: (n, 1)
    expected_profit = np.sum(weighted_price, axis=1, keepdims=True)

    return expected_profit
# 计算总成本矩阵
def calc_total_cost(price_per_unit, number_ls):
    price_per_unit = np.array(price_per_unit).reshape(1, -1)  # shape (1, 3)
    total_cost = np.sum(number_ls * price_per_unit, axis=1, keepdims=True)  # shape (n, 1)
    return total_cost
# 计算收益率矩阵
def calc_yield_rate(expected_profit, total_cost):
    # 避免除以 0
    with np.errstate(divide='ignore', invalid='ignore'):
        yield_rate = np.true_divide(expected_profit, total_cost)
        yield_rate[~np.isfinite(yield_rate)] = 0  # 将 inf 和 nan 转为 0
    return yield_rate
# 绘制收益率分布图
def plot_yield_distribution(yield_rates, threshold, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(yield_rates, kde=True, bins=30, color='skyblue')
    plt.axvline(threshold, color='red', linestyle='--', label=f'阈值: {threshold}')
    plt.title("收益率分布图")
    plt.xlabel("收益率")
    plt.ylabel("频数")
    plt.legend()
    plt.tight_layout()
    img_path = os.path.join(save_path, "收益率分布图.png")
    plt.savefig(img_path)
    plt.close()
    print(f"📈 收益率分布图已保存到：{img_path}")
# 计算组合索引
def find_combo_index(idx, index_ranges):
    for i, (start, end) in enumerate(index_ranges):
        if start <= idx < end:
            if i==0:
                return 0
            return i//len(all_combinations_num)
    raise IndexError(f"Index {idx} 不在任何组合范围内！")
# 保存结果到Excel
def save_results_to_excel(
    all_combination,
    skin_id_map_list,
    number_ls,
    Weight_ls,
    new_skin_itemfloat,
    converted_matrix,
    expected_profit,
    total_cost,
    yield_rate,
    secret_df,
    Superior,
    Confidentiality,
    skin_id_lz,
    threshold: float,
    itemfloat_avedge_dict,
    quality_dict,
    box_names,
    save_folder: str,
    save_name: str,
    n: int,
    plot_distribution=True,
    use_multithreading=True
):
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name.replace('.xlsx', '.csv'))

    combo_sample_index_ranges = []
    start = 0
    for arr in number_ls:
        count = arr.shape[0]
        combo_sample_index_ranges.append((start, start + count))
        start += count

    skin_name_map = Superior.set_index('skin_name_id')['skin_name'].to_dict()
    lz_skin_name_map = Confidentiality.set_index('skin_name_id')['skin_name'].to_dict()

    filtered_idx = np.where(yield_rate > threshold)[0]
    data_rows = []

    def process_single_index(idx):
        j = find_combo_index(idx, combo_sample_index_ranges)
        # 直接通过整除计算原始组合索引 j
        rows = []
        for j2 in range(len(skin_id_map_list[j])):
            for k in range(len(skin_id_map_list[j][j2])):
                sid = skin_id_map_list[j][j2][k]
                if sid == 0:
                    continue
                wear = new_skin_itemfloat[idx, j2, k]
                price = converted_matrix[idx, j2, k]
                label = get_wear_label(wear)
                row = []
                # 将组合中的每个箱子ID转换为名称
                original_comb = all_combination[j]
                converted_comb = [lookup_box.get(box_id, f"ID_{box_id}") for box_id in original_comb]
                row.extend(converted_comb)
                row.append(skin_name_map.get(sid, f"ID_{sid}"))
                row.append(round(price, 2))
                row.append(round(expected_profit[idx], 2))
                row.append(round(total_cost[idx], 2))
                row.append(round(yield_rate[idx], 4))
                found = False
                for box_index in range(n):
                    if sid in skin_id_map_list[j][box_index]:
                        box_weight = Weight_ls[idx][box_index]
                        box_skin_count = len(skin_id_map_list[j][box_index])
                        skin_prob = box_weight / box_skin_count if box_skin_count else 0
                        found = True
                        break
                if not found:
                    skin_prob = 0
                row.append(round(skin_prob, 6))
                row.extend(list(number_ls[idx]))
                row.extend([label, round(wear, 6)])
                for i in range(n):
                    lz_id = skin_id_lz[j][i]
                    lz_name = lz_skin_name_map.get(lz_id, f"ID_{lz_id}")
                    lz_wear = itemfloat_avedge_dict.get(lz_id, 0)
                    lz_price = quality_dict.get(lz_id, 0)
                    row.extend([lz_name, round(lz_wear, 6), round(lz_price, 2)])
                rows.append(row)
        return rows

    if use_multithreading:
        thread_count = get_thread_count(len(filtered_idx))
        print(f"🧵 构造结果行：使用 {thread_count} 个线程处理 {len(filtered_idx)} 条数据")

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            all_results = list(tqdm(
                executor.map(process_single_index, filtered_idx),
                total=len(filtered_idx),
                desc="多线程构造结果行"
            ))
        for r in all_results:
            data_rows.extend(r)
    else:
        for idx in tqdm(filtered_idx, desc="构造结果行"):
            data_rows.extend(process_single_index(idx))

    if not data_rows:
        print("⚠️ 没有满足条件的组合，未保存文件。")
        return

    all_columns = (
        [f"组合箱子{i+1}" for i in range(n)] +
        ["产物皮肤名称", "产物售价", "期望收益", "组合成本", "组合收益率"] +
        ["皮肤概率"] +
        [f"组合{i+1}_数量" for i in range(n)] +
        ["磨损区间", "磨损值"] +
        [
            label
            for i in range(n)
            for label in [
            f"炉渣{i + 1}_皮肤名称",
            f"炉渣{i + 1}_平均磨损",
            f"炉渣{i + 1}_均价"]
        ]
    )



    df = pd.DataFrame(data_rows, columns=all_columns)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 成功保存为 CSV 至：{save_path}")

    if plot_distribution:
        plot_yield_distribution(yield_rate, threshold, save_folder)



def pad_to_shape(tensor_list, target_k):
    """
    对 tensor_list 中每个形状为 (n, m, k_i) 的张量，在 axis=2 上 pad 到 target_k。
    """
    padded = []
    for t in tensor_list:
        n, m, k = t.shape
        if k == target_k:
            padded.append(t)
        else:
            # 构建 shape = (n, m, target_k - k) 的 0 矩阵
            pad_shape = (n, m, target_k - k)
            zeros = np.zeros(pad_shape)
            padded_tensor = np.concatenate([t, zeros], axis=2)
            padded.append(padded_tensor)
    return padded


def generate_partitions(n, k):
    results = []
    def backtrack(start, path):
        if len(path) == n - 1:
            last = k - sum(path)
            if last >= 0:
                results.append(tuple(sorted(path + [last])))
            return
        for i in range(k + 1):
            if sum(path) + i <= k:
                backtrack(start + 1, path + [i])
    backtrack(0, [])
    return sorted(set(results))



##################################### 主循环：多线程版本 ##########################################################################



def process_single_combination(i, n, Superior, secret, Confidentiality,
                               lookup_itemfloat_avedge, lookup_quality, all_combinations_num):
    try:
        i_ls = list(i)
        skin_id = [
            Superior[Superior['weapon_box_id'] == box]['skin_name_id'].tolist()
            for box in i_ls
        ]

        lookup_dict_nim = Superior.set_index('skin_name_id')['min_itemfloa'].to_dict()
        lookup_dict_max = Superior.set_index('skin_name_id')['max_itemfloa'].to_dict()
        new_skin_min = [[lookup_dict_nim.get(sid, 0) for sid in group] for group in skin_id]
        new_skin_max = [[lookup_dict_max.get(sid, 0) for sid in group] for group in skin_id]

        if any(len(sublist) == 0 for sublist in new_skin_min):
            return None

        max_len = max(len(sublist) for sublist in new_skin_min)
        new_skin_min = np.array([sublist + [0] * (max_len - len(sublist)) for sublist in new_skin_min])
        new_skin_max = np.array([sublist + [0] * (max_len - len(sublist)) for sublist in new_skin_max])

        skin_id_lz = [
            Confidentiality[Confidentiality['weapon_box_id'] == box]['skin_name_id'].tolist()[0]
            for box in i_ls
        ]
        itemfloat_avedge = np.array([lookup_itemfloat_avedge.get(sid, 0) for sid in skin_id_lz])
        quality_lz_ls = [lookup_quality.get(sid, 0) for sid in skin_id_lz]

        number_ls = np.array(generate_partitions(n, 10))
        Weight_ls = number_ls / 10
        new_skin_itemfloat = get_wear_values_by_skin_name(new_skin_max, new_skin_min, itemfloat_avedge, Weight_ls)
        converted_matrix = convert_wear_to_price_fast(new_skin_itemfloat, skin_id, secret)
        expected_profit = calc_expected_profit(converted_matrix, Weight_ls)
        total_cost = calc_total_cost(quality_lz_ls, number_ls)
        yield_rate = calc_yield_rate(expected_profit, total_cost)

        return {
            "comb": i_ls,
            "skin_id": skin_id,
            "skin_id_lz": skin_id_lz,
            "number_ls": number_ls,
            "Weight_ls": Weight_ls,
            "itemfloat": new_skin_itemfloat,
            "converted": converted_matrix,
            "profit": expected_profit,
            "cost": total_cost,
            "yield": yield_rate
        }

    except Exception as e:
        logging.error(f"❌ 处理组合 {i} 出错：{e}")
        return None


if __name__ == "__main__":

    # 配置日志记录器
    logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument('--n', type=int, default=3, help='箱子数量')
    parser.add_argument('--threshold', type=float, default=0.8, help='保存阈值')
    # 箱子数量
    n = int(parser.parse_args().n)

    # 保存阈值
    threshold = float(parser.parse_args().threshold)

    # 读取文件
    box_fill = parser.parse_args().input
    folder = parser.parse_args().output
    itemfloat_fill_path = "data/饰品磨损区间.xlsx"
    skin_file = "data/cs饰品编号/对应编号.xlsx"
    # 存储路径
    today = datetime.now().strftime("%Y-%m-%d")

    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, f"analysis_results_{today}.xlsx")
############################### 数据预处理 ###############################################################################
    # 导入磨损数据
    itemfloat_fill = pd.read_excel(itemfloat_fill_path)
    # 价格处理会将box_fill, skin_file中有用的数据列提取出来
    box_fill = 价格处理.main(box_fill, skin_file)

    # 预处理将数据分为三个部分 Superior是高品质皮肤不包含多个磨损区间，secret是高品质皮肤包含多个磨损，Confidentiality是只保留每个武器箱价格最低的低品质皮肤
    (Superior, secret, Confidentiality) = 模拟前预处理.clean_box(box_fill, itemfloat_fill)
    print(Superior.head())
    print(secret.head())
    print(Confidentiality.head())
    # 从Confidentiality中获取所有的箱子名称
    box_names = Confidentiality['weapon_box_id'].unique()
    # 获取所有箱子的组合并转化为列表
    all_combination = list(itertools.combinations(box_names, n))
##################################### 主循环 ##########################################################################
    all_data_rows = []
    all_combinations_num = generate_partitions(n, 10)
    lookup_itemfloat_avedge = Confidentiality.set_index('skin_name_id')['itemfloat'].to_dict()
    lookup_quality = Confidentiality.set_index('skin_name_id')['price'].to_dict()
    lookup_box = Confidentiality.set_index('weapon_box_id')['weapon_box'].to_dict()

    # 初始化结果列表
    your_skin_id_list = []
    all_combination_used = []
    all_skin_id_lz = []
    all_number_ls = []
    all_Weight_ls = []
    all_itemfloat_matrix = []
    all_converted_matrix = []
    all_expected_profit = []
    all_total_cost = []
    all_yield_rate = []

    # 多线程执行
    thread_count = get_thread_count(len(all_combination))
    print(f"🧵 正在使用 {thread_count} 个进程处理 {len(all_combination)} 个组合...")


    func = partial(
        process_single_combination,
        n=n,
        Superior=Superior,
        secret=secret,
        Confidentiality=Confidentiality,
        lookup_itemfloat_avedge=lookup_itemfloat_avedge,
        lookup_quality=lookup_quality,
        all_combinations_num=all_combinations_num
    )

    with ProcessPoolExecutor(max_workers=thread_count) as executor:
        results = list(tqdm(
            executor.map(func, all_combination),
            total=len(all_combination),
            desc="多进程处理中..."
        ))

    for result in results:
        if result is None:
            continue
        all_combination_used.append(result["comb"])
        your_skin_id_list.append(result["skin_id"])
        all_skin_id_lz.append(result["skin_id_lz"])
        all_number_ls.append(result["number_ls"])
        all_Weight_ls.append(result["Weight_ls"])
        all_itemfloat_matrix.append(result["itemfloat"])
        all_converted_matrix.append(result["converted"])
        all_expected_profit.append(result["profit"])
        all_total_cost.append(result["cost"])
        all_yield_rate.append(result["yield"])

    # 补齐 shape 的维度
    max_k = max(arr.shape[2] for arr in all_converted_matrix)
    converted_matrix = np.concatenate(pad_to_shape(all_converted_matrix, max_k), axis=0)
    new_skin_itemfloat = np.concatenate(pad_to_shape(all_itemfloat_matrix, max_k), axis=0)

    # 拼接向量值
    yield_rate = np.vstack(all_yield_rate).flatten()
    expected_profit = np.vstack(all_expected_profit).flatten()
    total_cost = np.vstack(all_total_cost).flatten()
    number_ls = np.vstack(all_number_ls)
    Weight_ls = np.vstack(all_Weight_ls)

    save_results_to_excel(
        all_combination=all_combination_used,  # 每组组合编号（84个）
        skin_id_map_list=your_skin_id_list,  # 每组组合对应的 skin_id（84个列表）
        number_ls=number_ls,  # (N_total, n)
        Weight_ls=Weight_ls,  # (N_total, n)
        new_skin_itemfloat=new_skin_itemfloat,  # (N_total, m, k)
        converted_matrix=converted_matrix,  # (N_total, m, k)
        expected_profit=expected_profit,  # (N_total,)
        total_cost=total_cost,  # (N_total,)
        yield_rate=yield_rate,  # (N_total,)
        secret_df=secret,  # secret 表
        Superior=Superior,  # 来源皮肤表
        Confidentiality=Confidentiality,  # 炉渣皮肤表
        skin_id_lz=all_skin_id_lz,  # 炉渣 ID 列表（84组）
        threshold=threshold,  # 你的阈值 float
        itemfloat_avedge_dict=lookup_itemfloat_avedge,  # 炉渣 id → 磨损 dict
        quality_dict=lookup_quality,  # 炉渣 id → price dict
        box_names = lookup_box,
        save_folder=folder,  # 保存目录
        save_name=f"analysis_results_{today}.xlsx",  # 保存文件名
        n=n,  # 箱子数量
        plot_distribution=True  # 是否画收益率分布图
    )



