from sklearn.datasets import load_wine
import pandas as pd

# 加载数据
data = load_wine()

# 转为 DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 添加标签列

# 保存为 CSV 文件
df.to_csv("wine_dataset.csv", index=False)

print("保存成功：wine_dataset.csv")
