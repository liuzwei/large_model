# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

np.set_printoptions(threshold=np.inf)  # numpy取消截断
pd.set_option('display.max_rows', None)  # pandas显示所有行
pd.set_option('display.max_columns', None)  # pandas显示所有列


# 声明两个变量地址，测试集和训练集地址
train_path = "C:\\develop\\aie53\\d5\\train_V2.csv"

debug = False
if debug == True:
    df_train = pd.read_csv(train_path, nrows=2000000)
else:
    df_train = pd.read_csv(train_path)

# 查看数据集的前5行
# print(df_train.head())

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

train = reduce_mem_usage(df_train)
# df_test = reduce_mem_usage(df_test)

# 删除排名为空的行
train.dropna(subset=['winPlacePerc'], inplace=True)
# df_train.drop(2744604, inplace=True)
# 查找缺失值
print(train[train['winPlacePerc'].isnull()])

# 每一局的人数
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')


# 归一化处理
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
# Compare standard features and normalized features
to_show = ['Id', 'kills','killsNorm','damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm', 'matchDuration', 'matchDurationNorm']
print(train[to_show].head(10))

# 创建新的特征，使用药品数量和使用道具数量加和
train['healsandboosts'] = train['heals'] + train['boosts']
print(train[['heals', 'boosts', 'healsandboosts']].tail())

# 移动总距离 = 行走距离 + 游泳距离 + 行驶距离
train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
# 无移动杀人数 = 杀人数 > 0 and 移动距离 = 0
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

# 爆头率
train['headshot_rate'] = train['headshotKills'] / train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)

# 异常值检测和剔除
# display(train[train['killsWithoutMoving'] == True].shape)
# print(train[train['killsWithoutMoving'] == True].head(10))
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
train.drop(train[train['roadKills'] > 10].index, inplace=True)

# 绘制击杀数和人数的分布情况
# plt.figure(figsize=(12,4))
# sns.countplot(data=train, x=train['kills']).set_title('Kills')

# 删除击杀大于40的行
train.drop(train[train['kills'] > 40].index, inplace=True)

# 绘制爆头率分布
# plt.figure(figsize=(12,4))
# sns.histplot(train['headshot_rate'], bins=10, kde=True)

# 爆头率为1且击杀数大于9的人数
display(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].shape)

# 绘制最远击杀距离分布
# plt.figure(figsize=(12,4))
# sns.histplot(train['longestKill'], bins=10, kde=True)

# 击杀距离大于1000的行数
display(train[train['longestKill'] >= 1000].shape)

# 删除最远击杀距离大于1000的行
train.drop(train[train['longestKill'] >= 1000].index, inplace=True)

print(train[['walkDistance', 'rideDistance', 'swimDistance', 'totalDistance']].describe())

# 绘制行走距离分布
# plt.figure(figsize=(12,4))
# sns.histplot(train['walkDistance'], bins=10, kde=True)
# 删除行走距离大于20000的行
train.drop(train[train['walkDistance'] >= 20000].index, inplace=True)

# 绘制骑行距离分布
# plt.figure(figsize=(12,4))
# sns.histplot(train['rideDistance'], bins=10, kde=True)
# 删除骑行距离大于30000的行
train.drop(train[train['rideDistance'] >= 30000].index, inplace=True)

# 绘制游泳距离分布
# plt.figure(figsize=(12,4))
# sns.histplot(train['swimDistance'], bins=10, kde=True)
# 删除游泳距离大于5000的行
train.drop(train[train['swimDistance'] >= 5000].index, inplace=True)

# 绘制武器分布
# plt.figure(figsize=(12,4))
# sns.countplot(data=train, x=train['weaponsAcquired']).set_title('Weapons Acquired')
# 删除武器数量大于80的行
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)

# 绘制治疗物品分布
# plt.figure(figsize=(12,4))
# sns.histplot(train['heals'], bins=10, kde=True)

# 删除治疗物品数量大于50的行
train.drop(train[train['heals'] >= 50].index, inplace=True)

# plt.show()

# print(train.shape)

print('在数据集中一共有 {} 种不同的匹配类型。'.format(train['matchType'].nunique()))

# 将 matchType 列（比赛类型）进行独热编码
train = pd.get_dummies(train, columns=['matchType'])

# 查看编码后的数据
matchType_encoding = train.filter(regex='matchType')
# print(matchType_encoding.head())

# 类型转换
train['groupId'] = train['groupId'].astype('category')
train['matchId'] = train['matchId'].astype('category')

# 将类别变量转换为数值型变量
train['groupId_cat'] = train['groupId'].cat.codes
train['matchId_cat'] = train['matchId'].cat.codes

# 删除原始的类别变量
train.drop(columns=['groupId', 'matchId'], inplace=True)

# 查看转换后的数据
# print(train[['groupId_cat', 'matchId_cat']].head())

# 删除ID列
train.drop(columns = ['Id'], inplace=True)

# 拆分数据，分为训练集和测试集
sample = 1000000
df_sample = train.sample(sample)
df = df_sample.drop(columns = ['winPlacePerc']) 
y = df_sample['winPlacePerc'] 
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# 模型训练
model = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',n_jobs=-1)
# n_jobs=-1 表示训练的时候，并行数和cpu的核数一样，如果传入具体的值，表示用几个核去跑

model.fit(X_train, y_train)

# 模型评分
y_pre = model.predict(X_test)
pre_score = model.score(X_test, y_test)
print('pre_score:', pre_score)

# mae评估
mae_result = mean_absolute_error(y_true=y_test, y_pred=y_pre)
print('mae_result:', mae_result)