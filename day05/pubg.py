import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# 声明两个变量地址，测试集和训练集地址
train_path = "C:/develop/aie53/d5/train_V2.csv"
test_path = "C:/develop/aie53/d5/test_V2.csv"

debug = True
if debug == True:
    df_train = pd.read_csv(train_path, nrows=100000)
    df_test  = pd.read_csv(test_path, nrows=1000)
else:
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

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

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)

# 杀敌数
temp_df = df_train['kills'].value_counts().sort_values(ascending=False)
fig_kill = plt.figure(figsize=(10, 6))
plt.title('kills count')
plt.xlabel('kills')
plt.ylabel('count')
sns.barplot(x=temp_df.index, y=temp_df)
# 柱状图上显示数量
for i, v in enumerate(temp_df):
    plt.text(i, v, v, ha='center', va='bottom', fontsize=5)

# 总伤害
data = df_train.copy()
data = data[data['kills']==0]
plt.figure(figsize=(10,6))
plt.title("Damage Dealt by 0 killers",fontsize=15)
sns.histplot(data['damageDealt'], kde=True)

# 胜率和击杀数的关系
sns.jointplot(x="winPlacePerc", y="kills", data=data, height=10, ratio=3, color="r")

# 根据击杀数（0 杀、1-2 杀、3-5 杀、6-10 杀和 10+ 杀）对玩家进行分组。
kills = df_train.copy()
# 分箱
kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])

plt.figure(figsize=(10, 6))
sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)

# 行走距离
data = df_train.copy()
data = data[data['walkDistance'] < df_train['walkDistance'].quantile(0.99)]
plt.figure(figsize=(10,6))
plt.title("Walking Distance Distribution",fontsize=15)
sns.histplot(data['walkDistance'], kde=True)
# 胜率和行走距离的关系
sns.jointplot(x="winPlacePerc", y="walkDistance",  data=data, height=10, ratio=3, color="lime")

# 驾车距离分布
data = df_train.copy()
data = data[data['rideDistance'] < df_train['rideDistance'].quantile(0.9)]
plt.figure(figsize=(10,6))
plt.title("Ride Distance Distribution",fontsize=15)
sns.histplot(data['rideDistance'], kde=True)
# 胜率和驾车距离的关系
sns.jointplot(x="winPlacePerc", y="rideDistance", data=data, height=10, ratio=3, color="y")

# 胜率和摧毁车辆数的关系
f,ax1 = plt.subplots(figsize =(10,6))
sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=data,color='#606060',alpha=0.8)
plt.xlabel('Number of Vehicle Destroys',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Vehicle Destroys/ Win Ratio',fontsize = 20,color='blue')
plt.grid()

# 治疗和提升物品
data = df_train.copy()
data = data[data['heals'] < data['heals'].quantile(0.99)]
data = data[data['boosts'] < data['boosts'].quantile(0.99)]

f,ax1 = plt.subplots(figsize =(10,6))
sns.pointplot(x='heals',y='winPlacePerc',data=data,color='lime',alpha=0.8)
sns.pointplot(x='boosts',y='winPlacePerc',data=data,color='blue',alpha=0.8)
plt.text(4,0.6,'Heals',color='lime',fontsize = 17,style = 'italic')
plt.text(4,0.55,'Boosts',color='blue',fontsize = 17,style = 'italic')
plt.xlabel('Number of heal/boost items',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Heals vs Boosts',fontsize = 20,color='blue')
plt.grid()
# 胜率和治疗数的关系
sns.jointplot(x="winPlacePerc", y="heals", data=data, height=10, ratio=3, color="lime")
# 胜率和提升物品数的关系
sns.jointplot(x="winPlacePerc", y="boosts", data=data, height=10, ratio=3, color="blue")



# 助攻数
temp_df = df_train['assists'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.title('assists count')
plt.xlabel('assists')
plt.ylabel('count')
sns.barplot(x=temp_df.index, y=temp_df)
# 柱状图上显示数量
for i, v in enumerate(temp_df):
    plt.text(i, v, v, ha='center', va='bottom', fontsize=5)


plt.show()



