import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier


# 加载数据
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'train.csv')
data = pd.read_csv(file_path)
# print(data.describe())

# 选择特征和标签
x = data[["Pclass", "Age", "Sex"]]
y = data["Survived"]

# 缺失值处理
# Age列缺失值处理
x["Age"] = x["Age"].fillna(x["Age"].mean())
# print(x.head())

# 数据划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

x_train = x_train.to_dict(orient='records')
x_test = x_test.to_dict(orient='records')

transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# print(x_test)

# 机器学习，决策树
estimator = DecisionTreeClassifier(max_depth=15)
# 训练模型
estimator.fit(x_train, y_train)

# 模型评估
y_pre = estimator.predict(x_test)
print(y_pre[:10])

ret = estimator.score(x_test, y_test)
print("准确率为：{}".format(ret))
