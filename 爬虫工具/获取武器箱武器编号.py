import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
from urllib.parse import quote
import re
# **从 cookie.txt 读取 Cookie**
with open("cookie.txt", "r", encoding="utf-8") as f:
    cookie_str = f.read().strip()

# **转换成字典**
cookies = {i.split("=")[0]: i.split("=")[1] for i in cookie_str.split("; ")}

# **请求头**
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://buff.163.com/"
}

# **手动输入武器编号**
weapon_ids = input("请输入武器编号（多个编号用逗号分隔）：").split(",")
weapon_case_name = "未知文件名"
# **存储数据**
skins_data = []

# **遍历每个武器编号**
for skin_id in weapon_ids:
    skin_id = skin_id.strip()  # 清除多余空格
    if not skin_id.isdigit():
        print(f"无效编号：{skin_id}")
        continue

    # **获取 container**
    container_api_url = f"https://buff.163.com/api/market/csgo_goods_containers?goods_id={skin_id}"
    response = requests.get(container_api_url, headers=HEADERS, cookies=cookies)

    if response.status_code != 200:
        print(f"获取 container 失败，状态码：{response.status_code}")
        continue

    try:
        container_data = response.json()
        container_name = container_data["data"]["containers"][0]["container"]
        container_name_encoded = quote(container_name)  # 对 container_name 进行 URL 编码
        container_type = container_data["data"]["container_type"]
        container_type_name = container_type  # 直接使用 container_type 作为 container_type_name

    except (KeyError, IndexError, json.JSONDecodeError):
        print(f"无法获取 container，跳过 {skin_id}")
        continue

    # **动态 API 请求**
    api_url = f"https://buff.163.com/api/market/csgo_container?container={container_name_encoded}&is_container=0&container_type={container_type_name}"
    response = requests.get(api_url, headers=HEADERS, cookies=cookies)


    if response.status_code != 200:
        print(f"API 请求失败，状态码：{response.status_code}")
        continue

    try:
        api_data = response.json()
        items = api_data["data"]["items"]
        weapon_case_name = api_data["data"]["container"]["name"]

    except (KeyError, json.JSONDecodeError):
        print(f"无法解析 API 返回数据，跳过 {skin_id}")
        continue

    # **遍历所有物品**
    for item in items:
        goods_id = item.get("goods_id")
        if not goods_id:
            continue

        # **访问武器详情页**
        skin_url = f"https://buff.163.com/goods/{goods_id}"
        page_response = requests.get(skin_url, headers=HEADERS, cookies=cookies)

        if page_response.status_code != 200:
            print(f"无法访问 {skin_url}，状态码：{page_response.status_code}")
            continue

        # **解析 HTML**
        soup = BeautifulSoup(page_response.text, "html.parser")

        # **获取皮肤名称**
        skin_name = soup.find("h1").text.strip() if soup.find("h1") else f"未知皮肤 ({goods_id})"

        # **获取品质信息**
        quality = "未知品质"
        quality_div = soup.find("div", class_="detail-cont")
        if quality_div:
            for span in quality_div.find_all("span"):
                if "品质" in span.text:
                    quality = span.text.replace("品质 |", "").strip()
                    break

        # **获取不同磨损状态对应的编号**
        wear_data = []
        stattrak_id = None  # 存储 StatTrak™ 的编号
        relative_goods_div = soup.find("div", {"class": "relative-goods"})
        if relative_goods_div:
            for wear in relative_goods_div.find_all("a", {"class": "i_Btn"}):
                wear_name = wear.text.strip().split(" ¥")[0]  # 磨损状态
                wear_id = wear.get("data-goodsid")
                is_stattrak = 0  # 是否StatTrak™

                if "StatTrak™" in wear.text:
                    is_stattrak = 1
                    stattrak_id = wear_id  # 存储 StatTrak™ 版本的编号

                if wear_id:
                    wear_data.append((wear_name, wear_id, is_stattrak))

        # **存储普通皮肤数据**
        for wear_name, wear_id, is_stattrak in wear_data:
            skins_data.append([skin_name, quality, wear_name, wear_id, is_stattrak])

        # **如果存在 StatTrak™，需要再次访问该页面获取全部磨损**
        if stattrak_id:
            stattrak_url = f"https://buff.163.com/goods/{stattrak_id}"
            stattrak_response = requests.get(stattrak_url, headers=HEADERS, cookies=cookies)

            if stattrak_response.status_code == 200:
                stattrak_soup = BeautifulSoup(stattrak_response.text, "html.parser")
                stattrak_goods_div = stattrak_soup.find("div", {"class": "relative-goods"})
                if stattrak_goods_div:
                    for wear in stattrak_goods_div.find_all("a", {"class": "i_Btn"}):
                        wear_name = wear.text.strip().split(" ¥")[0]
                        wear_id = wear.get("data-goodsid")
                        if wear_id:
                            skins_data.append([skin_name, quality, wear_name, wear_id, 1])  # 1 表示 StatTrak™

        # **延迟防封**
        time.sleep(1)

# 定义磨损优先级
wear_priority = ['崭新出厂', '略有磨损', '久经沙场', '破损不堪', '战痕累累']

def remove_brackets(text):
    # 使用正则表达式去除括号及其内容
    return re.sub(r'\([^)]*\)', '', text).strip()

# 定义一个函数，用于调整磨损值
# 定义磨损等级，从低到高（索引越大，磨损越重）



# 处理磨损的函数
def adjust_wear(df):
    # 按 "皮肤名称" 和 "是否StatTrak™" 进行分组
    for (skin, stattrak), group in df.groupby(['皮肤名称', '是否StatTrak™']):
        # 找到该组中已有的合法磨损等级（去掉 "普通" 和 "StatTrak™"）
        existing_wear = [w for w in group['磨损'] if w in wear_priority]

        if not existing_wear:
            continue  # 如果没有找到有效的磨损数据，跳过

        # 找到该组中最低的磨损等级
        min_wear = min(existing_wear, key=lambda x: wear_priority.index(x))

        # 计算上一级磨损（索引减一）
        new_wear_index = wear_priority.index(min_wear) - 1
        if new_wear_index >= 0:
            new_wear = wear_priority[new_wear_index]
        else:
            new_wear = wear_priority[0]  # 最小值就是 "崭新出厂"

        # **修改 "普通" 仅适用于 `是否StatTrak™ == 0`**
        df.loc[(df['皮肤名称'] == skin) & (df['是否StatTrak™'] == stattrak) &
               (df['磨损'] == '普通') & (stattrak == 0), '磨损'] = new_wear

        # **修改 "StatTrak™" 仅适用于 `是否StatTrak™ == 1`**
        df.loc[(df['皮肤名称'] == skin) & (df['是否StatTrak™'] == stattrak) &
               (df['磨损'] == 'StatTrak™') & (stattrak == 1), '磨损'] = new_wear

    return df


#数据整理
df = pd.DataFrame(skins_data, columns=["皮肤名称", "品质", "磨损", "编号", "是否StatTrak™"])
df['皮肤名称'] = df['皮肤名称'].apply(remove_brackets)
df.loc[df['磨损'] == '普通', '是否StatTrak™'] = 0
# 运行修正函数
df = adjust_wear(df)
# **保存到 Excel**
df.to_excel(rf"C:\Users\86150\Desktop\cs饰品\wuqixiang\{weapon_case_name}.xlsx", index=False)
print(f"数据已保存至 {weapon_case_name}.xlsx")
