# main_runner.py
import subprocess
import 数据预选
from datetime import datetime
import time
import math
from scipy.special import comb
##############################需要修改部分###########################################################################################
#数据保存路径
#使用前请确保路径存在
fell = "C:/Users/86150/Desktop/cs饰品"
#选择你要合成皮肤的高品质和对应低一级的品质以及炉渣的磨损,最后是输入箱子的数量和是否为StatTrak™，可看dat/data/cs饰品编号/cs饰品编号.xlsx.选择武器箱范围最多有42个武器箱数据
# {隐秘，保密，受限，军规级，工业级，消费级，崭新出厂，略有磨损，久经沙场，破损不堪，战痕累累,StatTrak™=1,0}
#如果最后一项是1，则表示模拟的数据为StatTrak™武器，否则为普通武器
skin_lst = ['保密', '受限', "崭新出厂",[0,42],0]
# 配置命令行参数 主要是模拟箱子数量（n） 和保存的阈值(threshold)
# 箱子数量是 3 （输入范围2-10） 表示程序会生成以获取箱子中3个箱子组合的所有可能，数值越大，组合数量越多
# 保存阈值 0.8 （输入范围是>0）表示最后保存的数据是收益率大于0.8的数据,算上平台抽成，需要1.03以上
n=4
threshold=1.03
#还需修改自己的cookie
##############################不需要调整######################################################################################################

# 按顺序配置脚本列表
scripts = [
    "mainexcel.py",
    "多元模拟.py",
]

def predict_computation(n: int, box_range: list) -> dict:
    """
    预测计算次数和运行时间。
    参数:
        n (int): 组合的箱子数量
        box_range (list): 箱子ID的范围，例如[0, 42]表示43个箱子
    返回:
        dict: 包含计算次数和时间预估的字典
    """
    # 计算组合数
    total_boxes = box_range[1] - box_range[0] + 1
    combinations = comb(total_boxes, n, exact=True)
    # 计算每个组合的分区数（将10个单位分配到n个箱子的分法）
    partitions = comb(10 + n - 1, n - 1, exact=True)
    # 总计算次数 = 组合数 × 分区数
    total_operations = combinations * partitions
    # 运行时间预估（基于基准假设：单核每秒处理5000次操作）
    # 并行加速：假设使用20进程（根据原代码的线程配置）
    base_speed = 5000  # 次/秒/核
    parallel_workers = 20
    time_seconds = total_operations / (base_speed * parallel_workers)
    #存储时间以构建每条数据0.005s,每个箱子有三个产物，达到阈值的数据按5%的比例来算
    time_write = total_operations*n*3*0.01*0.05/500
    time_seconds = time_seconds+time_write
    # 转换为小时和分钟
    hours = int(time_seconds // 3600)
    minutes = int((time_seconds % 3600) // 60)
    Second = int((time_seconds % 60))%60
    return {
        "组合数": combinations,
        "每个组合的分区数": partitions,
        "总计算次数": total_operations,
        "预估运行时间": f"{hours}小时{minutes}分钟{Second}秒",
        "假设条件": "基于20进程并行，每核每秒处5000次操作，构建每行数据0.01秒，平均产物为3，超过阈值占比5%，500条数据并行"
    }
result = predict_computation(n, skin_lst[3])

file_path = "data/cs饰品编号/cs饰品编号.xlsx"
today = datetime.now().strftime("%Y-%m-%d")
fell_excel= f"{fell}/数据/{today}_{skin_lst[0]}_{skin_lst[2]}_{skin_lst[4]}_sgo_items2.xlsx"
fell_old = f"{fell}/数据/"
output_path = f"{fell}/结果"

# config.py
PATHS = {
    "step1": {
        "input": "csgoItemsInfoFixed1.json",
        "output": fell_excel
    },
    "step2": {
        "input": fell_excel,
        "input1": fell_old,
        "input_ls":str(skin_lst),
        "output": output_path,
        '--n': str(n),
        '--threshold': str(threshold),
    },
}

serial_number=数据预选.main(file_path, skin_lst)
print(f"爬虫程序：本次共找到{serial_number}个编号,爬取预计需要{serial_number*1.25/60}分钟，如果今日已运行过程序会自动跳过")
print(f"模拟程序：本次运行将有{result['组合数']}个组合，将会进行{result['总计算次数']}次计算,预计需要{result['预估运行时间']},{result['假设条件']}")
print("预测时间仅供参考，并非真实运行时间！！！！")
Sure = input("是否还要进行,输入1继续:")
if Sure!='1':
    print("程序已退出")
    exit()

#开始计时
start_time = time.time()

print(f"正在运行函数{scripts[0]}")
subprocess.run([
    "python", scripts[0],
    "--input", PATHS["step1"]["input"],
    "--output", PATHS["step1"]["output"]
], check=True)
current_time = time.time()
print(f" {scripts[0]} 运行完成！用时{ (current_time-start_time)/60}")
print(f"正在运行函数{scripts[1]}")
print("此脚本运行过程中存在红色输出为正常现象，不要关闭程序！！！！")
for i in range(1,n+1):
    subprocess.run(
        ["python", scripts[1],
         "--input", PATHS["step2"]["input"],
         "--input1", PATHS["step2"]["input1"],
         "--output", PATHS["step2"]["output"],
         "--input_ls", PATHS["step2"]["input_ls"],
         "--n",str(i),
         "--threshold", PATHS["step2"]["--threshold"]],
        check=True)


end_time = time.time()
print(f" {scripts[1]} 运行完成！用时{(end_time-current_time)/60}分钟")
print(f"🎉 所有任务已完成！总用时{(end_time-start_time)/60}分钟")