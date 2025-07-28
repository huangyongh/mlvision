from sklearn.datasets import load_diabetes
import pandas as pd

# 加载数据
data = load_diabetes()

# 转为 DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 添加目标列

# 保存为 CSV（可选）
df.to_csv("diabetes_dataset.csv", index=False)
