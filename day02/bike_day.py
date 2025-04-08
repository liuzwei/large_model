import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 读取数据
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'bike-day.csv')
data = pd.read_csv(file_path)
# 去除临时用户数和注册用户数
data = data.drop(columns=['casual', 'registered'], axis=1)
# 打印数据的前5行
# print(data.head())

# 去除instant列、dteday列
data = data.drop(columns=['instant', 'dteday'], axis=1)
# print(data.info())

#  查看表头信息
# print(data.columns.values)

# 获取特征列
data_feature = data.drop(columns=['cnt'], axis=1)
data_label = data['cnt']
# print(data_feature.info())

# 对数据集进行划分
x_train, x_test, y_train, y_test  = train_test_split(data_feature, data_label, test_size=0.33, random_state=42)
print('x_train shape is {}'.format(x_train.shape))
print('y_train shape is {}'.format(y_train.shape))
print('-'*30)
print('x_test shape is {}'.format(x_test.shape))
print('y_test shape is {}'.format(y_test.shape))

#  初始化sklearn中线性回归模型为reg
reg = LinearRegression()
reg.fit(x_train, y_train)

predictions = reg.predict(x_test)
# print(predictions.shape)

predictions.flatten()
# print(predictions[:5])
# print(y_test.values)

# 数据可视化
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# 设置图像大小
plt.figure(figsize=(16, 6))
# 测试集：真实值
plt.plot(y_test.values, marker='.', label='actual',  linewidth=1.5)
# 测试集：预测值
plt.plot(predictions, marker='.', label='predictions', color='red', linewidth=2)

#图例位置
plt.legend(loc='best')
# plt.show()

# 使用评价函数计算精度
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# MAE法计算测试集与训练集预测精度
mae_lr = mean_absolute_error(y_test, predictions)
# MSE法计算测试集与训练集预测精度
mse_lr = mean_squared_error(y_test, predictions)

print('-'*30)
print('MAE: {}'.format(mae_lr))
print('MSE: {}'.format(mse_lr))

