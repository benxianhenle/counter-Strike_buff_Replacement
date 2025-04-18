import time
import requests
import json
import logging
import random
import pandas as pd
import argparse
import os
from tqdm import tqdm


'''

 * @author ELEVEN28th
 * @creat 2023-3-10

'''

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

data_list = []


def read_data(fell_json):
    with open(fell_json, 'r') as f:
        data = json.load(f)
    global goods_read
    goods_read = []
    for items in data:
        goods_read.append(items)


def buffcrawl_hi(fell_json,output):
    with open('cookie.txt', 'r') as f:
        cookie_str = f.read().strip()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0'
    }
    Cookie = {
        'Cookie': cookie_str
    }
    proxy = {'http': 'http://113.124.86.24:9999'}


    read_data(fell_json)
    #如果读取的excel文件存在，则读取，如果不存在，则创建
    if not os.path.exists(output):
        df_filtered = pd.DataFrame(columns=['goods_id', 'inspection_img_url', 'itemfloat', 'paintseed', 'price',
                                            'transaction_time', 'sticker_names', 'sticker_urls'])
        df_filtered.to_excel(output, index=False)
    df_filtered = pd.read_excel(output)
    #获取goods_id这一列
    goods_id = df_filtered['goods_id']
    orignal_url = 'https://buff.163.com/api/market/goods/bill_order?game=csgo&goods_id='

    # 准备DataFrame数据
    excel_data = []

    for i in tqdm(range(len(goods_read)), desc="Processing"):
        if i in goods_id:
            continue
        # 随机间隔（1-3秒）
        time.sleep(random.uniform(1, 1.5))

        all_id_url = orignal_url + str(goods_read[i])
        print(f"Processing: {goods_read[i]}")

        try:
            html_response = requests.get(url=all_id_url, headers=headers, cookies=Cookie, proxies=proxy)
            html_json = html_response.json()
            #print(html_json)
            html_items = html_json['data']['items']

        except Exception as e:
            logging.error(f"Error occurred while getting HTML response for url: {all_id_url}. {e}")
            continue

        for item in html_items:
            try:
                # 获取基础信息
                inspection_img_url = item['asset_info']['info'].get('inspect_url', '')
                itemfloat = item['asset_info']['paintwear']
                paintseed = item['asset_info']['info']['paintseed']
                price = item['price']
                transaction_time = item['transact_time']

                # 处理贴纸信息
                stickers = item['asset_info']['info'].get('stickers', [])
                sticker_names = []
                sticker_urls = []
                for sticker in stickers:
                    sticker_names.append(sticker.get('name', ''))
                    sticker_urls.append(sticker.get('img_url', ''))

                # 构建数据行
                row = {
                    'goods_id': goods_read[i],
                    'inspection_img_url': inspection_img_url,
                    'itemfloat': itemfloat,
                    'paintseed': paintseed,
                    'price': price,
                    'transaction_time': transaction_time,
                    'sticker_names': ', '.join(sticker_names),
                    'sticker_urls': ', '.join(sticker_urls)
                }
                excel_data.append(row)

            except Exception as e:
                logging.error(f"Error processing item: {e}")
                continue


    filename=output
    # 将数据加入excel,要保留原来的数据
    df_filtered = pd.read_excel(output)
    df_filtered = pd.concat([df_filtered, pd.DataFrame(excel_data)], ignore_index=True)
    df_filtered.to_excel(output, index=False)
    print(f"数据已保存到 {filename}")

# 配置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
fell_json = parser.parse_args().input
output = parser.parse_args().output

buffcrawl_hi(fell_json,output)