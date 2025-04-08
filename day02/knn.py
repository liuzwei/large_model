import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# 加载数据
data_raw = load_iris()
# 获取特征，标签数据 + 特征工程（标准化）
data_feature = StandardScaler().fit_transform(data_raw.data)
# 标签
data_label = data_raw.target

print(data_feature[:5])
print(data_label)
print(data_raw.target_names)

# 训练集、测试集数据分割
x_train, x_test, y_train, y_test = train_test_split(data_feature, data_label, test_size=0.3, random_state=34)
# 训练模型 + 交叉验证 + 网格搜索
estimator = KNeighborsClassifier()
# 需要搜索的参数
parameters = {
    # 这个就是knn的那个n
    'n_neighbors': [3,5,7,9,10,11,13,15,17]
}
# 3折交叉验证 + 网格搜索
estimator = GridSearchCV(estimator, param_grid=parameters, cv=3)

# 使用网格搜索得到的参数进行训练
estimator.fit(x_train, y_train)

# 预测
result = estimator.predict(x_test)
# 将预测结果和真实结果进行对比
compare_result = [i==j for i,j in zip(result, y_test)]
print(result)
print(y_test)
print(compare_result)

#  最佳参数是哪个
print(estimator.best_params_)
# 预测成功率是多少
print(estimator.score(x_test, y_test))