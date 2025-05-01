import pandas as pd
import itertools
import logging
import 模拟前预处理
import 价格处理
import argparse
# 配置日志记录器
logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
# 读取文件
itemfloat_fill_path = "data/饰品磨损区间.xlsx"
skin_file = "data/cs饰品编号/对应编号.xlsx"
#导入路径
box_fill = parser.parse_args().input
output_path = parser.parse_args().output

# 导入磨损数据
itemfloat_fill = pd.read_excel(itemfloat_fill_path)
# 价格处理会将box_fill, skin_file中有用的数据列提取出来
box_fill= 价格处理.main(box_fill, skin_file)
print(box_fill.head())
# 预处理将数据分为三个部分 Superior是高品质皮肤不包含多个磨损区间，secret是高品质皮肤包含多个磨损，Confidentiality是只保留每个武器箱价格最低的低品质皮肤
(Superior, secret, Confidentiality) = 模拟前预处理.clean_box(box_fill)


# 将secret中的皮肤名称转换为字符串类型，并去除前后空格,防止因为空隔导致的错误
itemfloat_fill['皮肤名称'] = itemfloat_fill['皮肤名称'].astype(str).str.strip()
secret['skin_name'] =secret['skin_name'].astype(str).str.strip()
secret['磨损']=secret['磨损'].astype(str).str.strip()


# 从Confidentiality中获取所有的箱子名称
box_names = Confidentiality['weapon_box'].unique()
# 获取所有箱子的组合并转化为列表
all_combination = list(itertools.combinations(box_names, 2))
#将 secret 按照weapon_box进行分组
grouped = Superior.groupby('weapon_box')

#################################################函数部分##################################################################
# 获取box_name中每个箱子对应的skin_name，以及在这个武器箱的概率,以及磨损极值
def get_skin_names(box_name):
    skin_names = []
    skin_itemfloat=[]
    # 获取box_name对应的skin_name
    for index, row in box_name.iterrows():
        skin_names.append(row['skin_name'])
        skin_itemfloat.append(get_wear_values_by_skin_name(row['skin_name']))
    box_Probability = 1 / len(box_name)
    return skin_names,box_Probability,skin_itemfloat

# 通过skin_name获取磨损极值
def get_wear_values_by_skin_name(skin):
    try:
        # 确保两边都是字符串，去除前后空格
        skin_clean = str(skin).strip()
        itemfloat_max = itemfloat_fill.loc[itemfloat_fill['皮肤名称'] == skin_clean, '最大磨损'].values[0]
        itemfloat_min = itemfloat_fill.loc[itemfloat_fill['皮肤名称'] == skin_clean, '最小磨损'].values[0]
    except IndexError:
        logging.error(f"IndexError: '皮肤名称' {skin} 不存在于 itemfloat_fill 中")
        print("未找到磨损皮肤：", skin)
        itemfloat_max=None
        itemfloat_min=None
    return (itemfloat_max,itemfloat_min)

#计算产物的磨损
def itemfloat_df(itemfloat,itemfloat_avedge):
    return (itemfloat[0] - itemfloat[1]) * itemfloat_avedge + itemfloat[1]

#计算收益率
def  calculate_expected_return(cost,expect):
     return expect/cost

#判断磨损区间
def itemfloat_if(itemfloat_num):
    if itemfloat_num<0.07:
        return "崭新出厂"
    elif itemfloat_num<0.15:
        return "略有磨损"
    elif itemfloat_num<0.38:
        return "久经沙场"
    elif itemfloat_num<0.45:
        return "破损不堪"
    else:
        return  "战痕累累"

#获取对应皮服对应磨损的价格
def get_price_by_wear(df, skin_name, wear_level):
    try:
        # 清洗数据
        skin_clean = str(skin_name).strip()
        wear_level_clean = str(wear_level).strip()
        price = df.loc[(df['skin_name'] == skin_clean) & (df['磨损'] == wear_level_clean), 'price'].values[0]
        return price, True  # True 表示是有效数据
    except IndexError:
        logging.error(f"IndexError: '皮肤名称' {skin_name} '磨损' {wear_level} 不存在于 df 中")
        return 1, False  # False 表示这是默认值


######################################################表格构建#############################################################
#创建一个空的pd表用于存放收益率，表头有“武器箱组合” “1：9”“2：8”“3：7”“4：6”“5：5”“6：4”“7：3”“8：2”“9：1”
yield_table = pd.DataFrame(columns=["武器箱组合","0:10", "1:9", "2:8", "3:7", "4:6", "5:5", "6:4", "7:3", "8:2", "9:1","10:0"])
#创建一个表用来存储成本
cost_table = pd.DataFrame(columns=["武器箱组合","0:10", "1:9", "2:8", "3:7", "4:6", "5:5", "6:4", "7:3", "8:2", "9:1","10:0"])
#创建一个表用来存储每一个武器的出货概率和价格
probability_table = pd.DataFrame(columns=["武器箱组合","皮肤名称","0:10", "1:9", "2:8", "3:7", "4:6", "5:5", "6:4", "7:3", "8:2", "9:1","10:0"])
#创建一个表用来存储汰换武器磨损
itemfloat_table= pd.DataFrame(columns=["武器箱组合","皮肤名称","0:10", "1:9", "2:8", "3:7", "4:6", "5:5", "6:4", "7:3", "8:2", "9:1","10:0"])
#创建一个表用来存储炉渣1信息
yield_price_table = pd.DataFrame(columns=["武器箱组合" , "皮肤名称","平均磨损","均价"])
#创建一个表用来存储炉渣2信息
yield_price_table_2 = pd.DataFrame(columns=["武器箱组合" , "皮肤名称","平均磨损","均价"])
#创建一个表用来存储缺失值
missing_price_table = pd.DataFrame(columns=["武器想组合" , "皮肤名称","磨损区间"])

#######################################################主体循环############################################################

idx=0
for box_name1, box_name2 in all_combination:

    # 创建一个列表，用来存储yield_table
    yield_table_ll=[box_name1+"|"+box_name2]
    #创建一个列表，用来存储cost_table
    cost_table_ll=[box_name1+"|"+box_name2]

    # 创建一个列表，用来存储probability_table
    probability_table_ll=[]
    # 创建一个列表，用来存储itemfloat_table
    itemfloat_table_ll = []
    # 创建一个列表，用来存储缺失值
    missing_price_records = []


    #在Confidentiality表中获取box_name1, box_name2箱子对应炉渣的价格
    price_1 = Confidentiality.loc[Confidentiality['weapon_box'] == box_name1, 'price'].values[0]
    price_2 = Confidentiality.loc[Confidentiality['weapon_box'] == box_name2, 'price'].values[0]
    # 在Confidentiality表中获取box_name1, box_name2箱子对应炉渣的名称
    skin_name_1 = Confidentiality.loc[Confidentiality['weapon_box'] == box_name1, 'skin_name'].values[0]
    skin_name_2 = Confidentiality.loc[Confidentiality['weapon_box'] == box_name2, 'skin_name'].values[0]
    # 获取box_name1, box_name2箱子对应炉渣的平均磨损
    itemfloat_avedge1 = Confidentiality.loc[Confidentiality['weapon_box'] == box_name1, 'itemfloat'].values[0]
    itemfloat_avedge2 = Confidentiality.loc[Confidentiality['weapon_box'] == box_name2, 'itemfloat'].values[0]

    # 创建一个列表，用来存储yield_price_table
    yield_price_ls_1=[box_name1+"|"+box_name2,skin_name_1,itemfloat_avedge1,price_1]
    yield_price_ls_2=[box_name1+"|"+box_name2,skin_name_2,itemfloat_avedge2,price_2]

    # 获取box_name1, box_name2在Superior对应的行
    Superior_1 = grouped.get_group(box_name1)
    Superior_2 = grouped.get_group(box_name2)
    #获取box_name1, box_name2箱子对应的皮肤名称，以及磨损极值，以及Superior中每个皮肤的概率
    skin_names1,box_Probability1,skin_itemfloat1 = get_skin_names(Superior_1)
    skin_names2,box_Probability2,skin_itemfloat2 = get_skin_names(Superior_2)

    #循环创建列表
    for i in range(len(skin_names1)):
        itemfloat_table_ll.append([box_name1+"|"+box_name2, skin_names1[i]])
        probability_table_ll.append([box_name1+"|"+box_name2, skin_names1[i]])
    for i in range(len(skin_names2)):
        itemfloat_table_ll.append([box_name1+"|"+box_name2, skin_names2[i]])
        probability_table_ll.append([box_name1+"|"+box_name2, skin_names2[i]])


    for i in range(11):
        if i==0:
            skin_p1=0
            skin_p2=1
        else:
            skin_p1 = i / 10
            skin_p2 = 1 - skin_p1

        itemfloat_avedge=itemfloat_avedge1*skin_p1+itemfloat_avedge2*skin_p2
        cost=price_1*i+price_2*(10-i)
        cost_table_ll.append(cost)
        expect = []
        valid=True
        for j in range(len(skin_itemfloat1)):
            itemfloat_ll = itemfloat_df(skin_itemfloat1[j],itemfloat_avedge)
            itemfloat_table_ll[j].append(itemfloat_ll)
            itemfloat_ll = itemfloat_if(itemfloat_ll)
            price_nv, valid_price = get_price_by_wear(secret, skin_names1[j], itemfloat_ll)
            if valid_price:
                expect.append(price_nv * box_Probability1 * skin_p1)
            else:
                expect.append(None)
                missing_price_records.append([box_name1+"|"+box_name2 , skin_names1[j], itemfloat_ll])
                valid = False
                logging.warning(
                    f"⚠️ 无法获取价格: {skin_names1[j]} 磨损: {itemfloat_ll} 组合: {box_name1}|{box_name2} 比例: {i}/10")

            probability_table_ll[j].append(box_Probability1*skin_p1)

        for j in range(len(skin_itemfloat2)):
            itemfloat_ll = itemfloat_df(skin_itemfloat2[j],itemfloat_avedge)
            itemfloat_table_ll[j + len(skin_itemfloat1)].append(itemfloat_ll)
            itemfloat_ll = itemfloat_if(itemfloat_ll)
            price_nv, valid_price=get_price_by_wear(secret, skin_names2[j], itemfloat_ll)
            if valid_price:
                expect.append(price_nv * box_Probability2 * skin_p2)
            else:
                expect.append(None)
                missing_price_records.append([box_name1+"|"+box_name2, skin_names2[j], itemfloat_ll])
                valid = False
                logging.warning(
                    f"⚠️ 无法获取价格: {skin_names2[j]} 磨损: {itemfloat_ll} 组合: {box_name1}|{box_name2} 比例: {i}/10")
            probability_table_ll[j+ len(skin_itemfloat1)].append(box_Probability2 * skin_p2)
        if valid:
            yield_table_ll.append(calculate_expected_return(cost,sum(expect)))
        else:
            yield_table_ll.append(None)


    #将yield_table_ll写入yield_table作为一行数据
    yield_table.loc[idx] = yield_table_ll
    cost_table.loc[idx] = cost_table_ll
    yield_price_table.loc[idx] = yield_price_ls_1
    yield_price_table_2.loc[idx] = yield_price_ls_2

    idx += 1

    probability_df = pd.DataFrame(probability_table_ll,
                                  columns=["武器箱组合","皮肤名称","0:10", "1:9", "2:8", "3:7", "4:6", "5:5", "6:4", "7:3", "8:2", "9:1","10:0"])
    itemfloat_tabel_df = pd.DataFrame(itemfloat_table_ll,
                                      columns=["武器箱组合", "皮肤名称", "0:10", "1:9", "2:8", "3:7", "4:6", "5:5", "6:4", "7:3", "8:2", "9:1","10:0"])
    price_tabel_df  = pd.DataFrame(missing_price_records, columns=["武器箱组合", "皮肤名称", "磨损区间"])
    if not probability_df.empty:
        probability_table = pd.concat([probability_table, probability_df], ignore_index=True)
        itemfloat_table = pd.concat([itemfloat_table, itemfloat_tabel_df], ignore_index=True)
        missing_price_table = pd.concat([missing_price_table, price_tabel_df], ignore_index=True)


# 对missing_price_table进行去重
missing_price_table = missing_price_table.drop_duplicates()
yield_price_table = yield_price_table.drop_duplicates()
yield_price_table_2 = yield_price_table_2.drop_duplicates()

# 将价格表加入
superior_extracted = secret[['skin_name', 'price', '磨损']]
# 使用 pivot 方法将磨损列的值作为新的表头，并重置索引保留skin_name列
# 使用 pivot_table 并指定聚合函数（例如取第一个出现的值）
superior_pivot = (
    superior_extracted
    .pivot_table(
        index='skin_name',
        columns='磨损',
        values='price',
        aggfunc='first'  # 可选：'mean', 'sum', 'max', 'min' 或自定义函数
    )
    .reset_index()  # 将索引转换为普通列
    .rename_axis(columns=None)  # 清除列名层级
    .fillna(0)  # 填充缺失值
)

# 验证结果
print(superior_pivot.head())


# 使用 ExcelWriter 保存多个 DataFrame
with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    yield_table.to_excel(writer, sheet_name="收益率", index=False)
    cost_table.to_excel(writer, sheet_name="成本", index=False)
    probability_table.to_excel(writer, sheet_name="出货概率", index=False)
    itemfloat_table.to_excel(writer, sheet_name="磨损", index=False)
    missing_price_table.to_excel(writer, sheet_name="缺失价格", index=False)
    yield_price_table.to_excel(writer, sheet_name="炉渣1", index=False)
    yield_price_table_2.to_excel(writer, sheet_name="炉渣2", index=False)
    superior_pivot.to_excel(writer, sheet_name="价格", index=False)

print("数据已成功保存到 Excel 文件！ 🎉")

