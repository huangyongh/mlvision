import pandas as pd

# 数据集 URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"

# 读取数据
df = pd.read_csv(url)

# 保存为本地文件
df.to_csv("pollution.csv", index=False)

print("已成功保存为 pollution.csv")
