import json
import requests
import pandas as pd
import time
#开始计时
start_time = time.time()
fell =r"C:\Users\86150\Desktop\cs饰品\磨损极值\饰品磨损区间.xlsx"
fell =pd.read_excel(fell)
skin_name_1 =fell.set_index("皮肤名称")
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
        #如果"skin_name在skin_name_1中则跳过此次循环
        if skin_name in skin_name_1.index:
            continue


        if not all([skin_name, goods_id]):
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

# === 6. 保存 JSON 和 Excel 文件 ===
with open("饰品磨损区间.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(processed_data)
df.to_excel("饰品磨损区间.xlsx", index=False)
#结束计时
end_time = time.time()
print(f"✅ 所有数据处理完成并已保存！此次用时{end_time-start_time}")
