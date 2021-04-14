import pandas as pd
import numpy as np
import sys


''' 全局参数设置 '''
data_path = "D:\\Kaggle\\DataSets\\accepted_2007_to_2018q4.csv\\accepted_2007_to_2018Q4.csv"
test_data_path = "D:\\Kaggle\\DataSets\\accepted_2007_to_2018q4.csv\\test.csv"
test_mode = True        # 为True时使用小数据集
delete_rate = 0.50  # 数据预处理接段如果超过该值则会剔除该行对应数据
enum_upper = 12*12  # 当一列中出现的值种类少于该值时认定为枚举类
ENUM_NUMBER = 0  # 枚举型整数数据
ENUM_STRING = 1  # 枚举型字符数据
NOT_ENUM_NUMBER = 2  # 非枚举型整型数据
NOT_ENUM_STRING = 3  # 非枚举型字符数据
UNKNOWN = 4  # 未知情况类型

''' 根据test_mode值修改其中相应参数 '''
if test_mode:
    enum_upper = 12

''' 从.csv中读入征信数据 '''
if test_mode:
    data = pd.read_csv(test_data_path)
else:
    data = pd.read_csv(data_path)
# print(data)  # 数据大小 [2260701 rows x 151 columns]
''' 统计行数和列数 '''
rows_count = data.shape[0]
cols_count = data.shape[1]

''' 数据清洗阶段 '''
''' 剔除缺失值超过0.5的数据 '''
bool_matrix = data.isnull().sum(axis=1)  # 获取每行nan数量
nan_in_row_countList = bool_matrix.values.tolist()  # 转化成list方便操作
line_to_delete = []  # 设置一个临时存放要删除行的list
for line in range(len(nan_in_row_countList)):
    ''' 计算第line行的缺失值 '''
    lose_rate = nan_in_row_countList[line] / cols_count
    if lose_rate > delete_rate:
        ''' 删除第line行 '''
        line_to_delete.append(line)
''' 行删除操作 '''
data = data.drop(index=line_to_delete)
''' 此过程之后删除所有值为nan的列 '''
data = data.dropna(axis=1, how="all")
''' 重新统计行数和列数 '''
rows_count = data.shape[0]
cols_count = data.shape[1]

''' 数据预处理接段 '''
''' 对每列数据进行4种分类 '''
cols_list = data.columns.tolist()
cols_info_dict = {}  # 设置一个字典，为 列名str:（类型，众数）
cols_dtype_list = data.dtypes.values.tolist()  # 获取每一列对应的默认dtype属性
# print(cols_dtype_list)
# print(cols_dtype_list[4] == np.dtype('object'))
''' 分成了np.dtype()——int64，float64，object '''
for col in cols_list:
    myType = 0
    avg_num = None
    mode_num = None
    values_list = None
    ''' 判断是否为枚举型 '''
    if len(data[col].unique()) >= enum_upper:
        ''' 非枚举类型对应二进制位第二位为1 '''
        myType += 2
    if data[col].dtype == np.dtype('int64') or data[col].dtype == np.dtype('float64'):
        myType += 0
    elif data[col].dtype == np.dtype('object'):
        myType += 1
    else:
        myType = 4
    if myType % 2 == 0:
        ''' 对于数型数据，统计其均值和众数 '''
        avg_num = data[col].mean()
        mode_num = data[col].mode()[0]
    if myType == 1:
        ''' 对于枚举型字符串，统计其众数 '''
        mode_num = data[col].mode()[0]
    if len(data[col].unique()) < enum_upper:
        ''' 补充一些对于枚举类型的记录 '''
        values_list = list(data[col].unique())
    cols_info_dict[col] = (myType, avg_num, mode_num, values_list)
''' 对类别进行统计后，进行插值，枚举型采用众数插值，非枚举型采用均值插值 '''
col_to_delete = []
for col in cols_list:
    myType, avg_num, mode_num, values_list = cols_info_dict[col]        # 元组解包
    if myType <= 1:
        data[col].fillna(value=mode_num, inplace=True)
    elif myType == 2:
        data[col].fillna(value=0, inplace=True)
    elif myType == 3:
        data[col].fillna(value='', inplace=True)
    else:
        ''' 准备去掉这一列 '''
        col_to_delete.append(col)
''' 列删除操作 '''
data = data.drop(columns=col_to_delete)
''' 重新统计行数和列数 '''
rows_count = data.shape[0]
cols_count = data.shape[1]
''' 重新获取列信息 '''
cols_list = data.columns.tolist()
''' 统计字典中删除该键值 '''
for col in col_to_delete:
    del cols_info_dict[col]



