import json
import requests
import pandas as pd
import time
import os

# 开始计时
start_time = time.time()

# === 0. 加载已有磨损数据（如存在） ===
if os.path.exists("饰品磨损区间StatTrak.xlsx"):
    fell = pd.read_excel("饰品磨损区间StatTrak.xlsx")
    skin_name_1 = fell.set_index("皮肤名称")
    skip_existing = True
    print("🟡 已加载已有磨损数据，将跳过已处理皮肤")
else:
    skin_name_1 = pd.DataFrame(columns=["最小磨损", "最大磨损"])  # 空 DataFrame，防止索引错误
    skip_existing = False
    print("🟢 未找到已处理数据，将处理全部皮肤")

# === 1. 从 cookie.txt 读取 Cookie 并转换为字典 ===
with open("cookie.txt", "r", encoding="utf-8") as f:
    cookie_str = f.read().strip()

cookies = {i.split("=")[0]: i.split("=")[1] for i in cookie_str.split("; ")}

# === 2. 请求头设置 ===
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://buff.163.com/"
}

# === 3. 读取编号文件 ===
with open("cs饰品编号.json", "r", encoding="utf-8") as f:
    goods_data = json.load(f)

processed_data = []

# === 4. 获取 paintwear 的函数（新版 URL 逻辑）===
def get_paintwear(goods_id, is_max=False):
    base_url = "https://buff.163.com/api/market/paintwear_rank"
    params = {
        "game": "csgo",
        "goods_id": goods_id,
        "page_num": 1,
        "rank_type": 0
    }
    if is_max:
        params["order_type"] = 1  # 最大磨损需要加上这个参数

    try:
        response = requests.get(base_url, headers=HEADERS, cookies=cookies, params=params, timeout=10)
        response.raise_for_status()
        res_json = response.json()
        return res_json.get("data", {}).get("ranks", [{}])[0].get("paintwear")
    except Exception as e:
        print(f"[!] 获取 goods_id={goods_id} is_max={is_max} 时出错：{e}")
        return None

# === 5. 遍历 JSON 数据 ===
for weapon_case, skins in goods_data.items():
    for skin in skins:
        skin_name = skin.get("skin_name")
        goods_id = skin.get("number")

        if not all([skin_name, goods_id]):
            continue

        # 跳过已处理的皮肤
        if skip_existing and skin_name in skin_name_1.index:
            print(f"⏭️ 已处理过，跳过：{skin_name}")
            continue

        print(f"🔍 正在处理：{skin_name}（{weapon_case}）")

        min_wear = get_paintwear(goods_id, is_max=False)
        time.sleep(1)
        max_wear = get_paintwear(goods_id, is_max=True)
        time.sleep(1)

        processed_data.append({
            "皮肤名称": skin_name,
            "武器箱": weapon_case,
            "最小磨损": min_wear,
            "最大磨损": max_wear
        })

# === 6. 合并已有数据（如存在）并保存 ===
if skip_existing:
    df_old = skin_name_1.reset_index()
    df_new = pd.DataFrame(processed_data)
    df = pd.concat([df_old, df_new], ignore_index=True)
else:
    df = pd.DataFrame(processed_data)

# 保存到文件
with open("饰品磨损区间StatTrak.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

df.to_excel("饰品磨损区间StatTrak.xlsx", index=False)

# 结束计时
end_time = time.time()
print(f"✅ 所有数据处理完成并已保存！此次用时 {end_time - start_time:.2f} 秒")
