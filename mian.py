# main_runner.py
import subprocess
import 数据预选
from datetime import datetime
import math
import time
##############################需要修改部分###########################################################################################
#数据保存路径
#使用前请确保路径存在
fell = "C:/Users/86150/Desktop/cs饰品"
#选择你要合成皮肤的高品质和对应低一级的品质以及炉渣的磨损,最后是输入箱子的数量，可看dat/data/cs饰品编号/cs饰品编号.xlsx.选择武器箱范围最多有42个武器箱数据
# {隐秘，保密，受限，军规级，工业级，消费级，崭新出厂，略有磨损，久经沙场，破损不堪，战痕累累}
skin_lst = ['保密', '受限', "崭新出厂",[0,42]]
# 配置命令行参数 主要是模拟箱子数量（n） 和保存的阈值(threshold)
# 箱子数量是 3 （输入范围2-10） 表示程序会生成以获取箱子中3个箱子组合的所有可能，数值越大，组合数量越多
# 保存阈值 0.8 （输入范围是0-1）表示最后保存的数据是收益率大于0.8的数据,
n=3
threshold=0.8
#还需修改自己的cookie
##############################不需要调整######################################################################################################

# 按顺序配置脚本列表
scripts = [
    "mainexcel.py",
    "多元模拟.py",
]


operations =  math.comb(skin_lst[3][1]-skin_lst[3][0]+1,n) * math.comb(10+n-1,n)


file_path = "data/cs饰品编号/cs饰品编号.xlsx"
today = datetime.now().strftime("%Y-%m-%d")
fell_excel= f"{fell}/数据/{today}_{skin_lst[0]}_{skin_lst[2]}_sgo_items2.xlsx"
output_path = f"{fell}/结果"

# config.py
PATHS = {
    "step1": {
        "input": "csgoItemsInfoFixed1.json",
        "output": fell_excel
    },
    "step2": {
        "input": fell_excel,
        "output": output_path,
        '--n': str(n),
        '--threshold': str(threshold),
    },
}

serial_number=数据预选.main(file_path, skin_lst)
print(f"本次共找到{serial_number}个编号,爬取预计需要{serial_number*1.25/60}分钟")
print(f"本次运行将有{operations}个组合，将会进行{operations*4}次计算")
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
subprocess.run(
    ["python", scripts[1],
    "--input", PATHS["step2"]["input"],
    "--output", PATHS["step2"]["output"],
    "--n", PATHS["step2"]["--n"],
    "--threshold", PATHS["step2"]["--threshold"]],
    check=True)

end_time = time.time()
print(f" {scripts[1]} 运行完成！用时{end_time-current_time}")
print(f"🎉 所有任务已完成！总用时{end_time-start_time}")