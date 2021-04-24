import pandas as pd
import numpy as np
import math
import methods
import settings as st

''' 全局参数设置 '''
data_path = st.data_path
test_data_path = st.test_data_path
test_mode = st.test_mode  # 为True时使用小数据集
delete_rate = st.delete_rate  # 数据预处理接段如果超过该值则会剔除该行对应数据
enum_upper = st.enum_upper  # 当一列中出现的值种类少于该值时认定为枚举类
ENUM_NUMBER = st.ENUM_NUMBER  # 枚举型整数数据
ENUM_STRING = st.ENUM_STRING  # 枚举型字符数据
NOT_ENUM_NUMBER = st.NOT_ENUM_NUMBER  # 非枚举型整型数据
NOT_ENUM_STRING = st.NOT_ENUM_STRING  # 非枚举型字符数据
UNKNOWN = st.UNKNOWN  # 未知情况类型
group_max_num = st.group_max_num  # 对于非枚举型变量分组中最多有几组
IV_critical = st.IV_critical  # 对于IV值低于IV_critical的属性进行剔除

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
    ''' 要剔除loan_status状态为Issued的数据 '''
    if lose_rate > delete_rate or data.loc[line, "loan_status"] == "Issued":
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
cols_info_dict = {}  # 设置一个字典，为 列名str:（类型，均值，众数，标准差，枚举值）
cols_dtype_list = data.dtypes.values.tolist()  # 获取每一列对应的默认dtype属性
# print(cols_dtype_list)
# print(cols_dtype_list[4] == np.dtype('object'))
''' 分成了np.dtype()——int64，float64，object '''
for col in cols_list:
    myType = 0
    avg_num = None
    mode_num = None
    std_num = None
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
        ''' 对于数型数据，统计其均值，众数和标准差 '''
        avg_num = data[col].mean()
        mode_num = data[col].mode()[0]
        std_num = data[col].std()
    if myType == 1:
        ''' 对于枚举型字符串，统计其众数 '''
        mode_num = data[col].mode()[0]
    if len(data[col].unique()) < enum_upper:
        ''' 补充一些对于枚举类型的记录 '''
        values_list = list(data[col].unique())
    cols_info_dict[col] = (myType, avg_num, mode_num, std_num, values_list)
''' 对类别进行统计后，进行插值，枚举型采用众数插值，非枚举型采用均值插值 '''
col_to_delete = []
for col in cols_list:
    myType, avg_num, mode_num, std_num, values_list = cols_info_dict[col]  # 元组解包
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

''' 对于每一列进行分组，计算WOE值和IV值 '''
''' 统计总共有多少正常客户或者违约客户 '''
yt = 0
nt = 0
for index, row in data.iterrows():
    if methods.is_default(row):
        yt += 1
    else:
        nt += 1
# print("yt={0}, nt={1}".format(yt, nt))
print("Calculating:")
storage = {}  # 用于生成log
for col in cols_list:
    print(col + ":")
    IV = 0.0
    if cols_info_dict[col][0] == 2:
        ''' 非枚举整型处理方法 '''
        group_volume = int(rows_count / group_max_num) + 1  # 计算每一组应该由多少数据
        ''' 根据这些列进行排序 '''
        data_copy = data.sort_values(col)
        data_col = list(data_copy[col].values)
        critical_col = list(data_copy["loan_status"].values)
        for i in range(group_max_num):
            ''' 划定区间为 [start_row, end_row) '''
            start_row = i * group_volume
            end_row = min((i + 1) * group_volume, len(data_col))
            # print(start_row, end_row)
            ''' 统计该分组中的yi和ni '''
            yi = 0
            ni = 0
            for j in range(start_row, end_row):
                if methods.value_is_default(critical_col[j]):
                    yi += 1
                else:
                    ni += 1
            ''' 如果该分组中全是一种客户，则该变量与关键属性强相关 '''
            if yi == 0 or ni == 0:
                IV = float("inf")
                break
            ''' 计算该分组的WOE值 '''
            pyi = yi / yt
            pni = ni / nt
            WOEi = np.log(pyi / pni)
            ''' 计算该分组的IV值，并将其加到该属性的IV值中 '''
            IVi = (pyi - pni) * WOEi
            IV += IVi
        print("IV : {0}".format(IV))

    if cols_info_dict[col][0] < 2:
        ''' 枚举型数据按值分组 '''
        col_statics_dict = {}
        ''' 新建一个字典，为 枚举值：[yi, ni] '''
        for value in cols_info_dict[col][-1]:
            col_statics_dict[value] = [0, 0]
        ''' 这个枚举值字典中会出现nan，原因是进行字典分析是在插值前，但对课题没有影响 '''
        ''' 进行行遍历 '''
        for index, row in data.iterrows():
            key = row[col]
            if methods.is_default(row):
                col_statics_dict[key][0] += 1
            else:
                col_statics_dict[key][1] += 1
        ''' 开始计算枚举值对应的WOE和IV '''
        for value in cols_info_dict[col][-1]:
            yi = col_statics_dict[value][0]
            ni = col_statics_dict[value][1]
            pyi = col_statics_dict[value][0] / yt
            pni = col_statics_dict[value][1] / nt
            ''' 如果该分组中全是一种客户，则该变量与关键属性强相关 '''
            if yi == 0 or ni == 0:
                IV = float("inf")
                break
            WOEi = np.log(pyi / pni)
            IVi = (pyi - pni) * WOEi
            IV += IVi
        print("IV : {0}".format(IV))

    if cols_info_dict[col][0] > 2:
        print("IV : 0.0")
    print(cols_info_dict[col])
    ''' 该列进行统计并记录 '''
    storage[col] = (cols_info_dict[col][0], IV, cols_info_dict[col][1], cols_info_dict[col][2], \
                    cols_info_dict[col][3], cols_info_dict[col][-1])

# print(storage)
''' 进行存储 '''
# methods.statics(cols_iterable=cols_list, storage_dict=storage)
''' 删除IV值低于0.30的属性 '''
temp_cols_list = cols_list[:]
for col in temp_cols_list:
    if storage[col][1] is None or storage[col][1] < IV_critical:
        del storage[col]
        cols_list.remove(col)
# print(len(cols_list))
''' 根据当前所要选的剩余的 '''
list_len = len(cols_list)
element_size = 32 * 32
while 1024 / element_size < list_len:
    element_size /= 4
print(element_size)
''' 构建该文件对应的存储灰度图文件夹 '''
pictures_path = methods.build()
''' 生成灰度图列表 '''
for index, row in data.iterrows():
    ''' 生成一个只有各个属性对应的一个灰度值的list '''
    gray_list = []
    for col in cols_list:
        if storage[col][0] == NOT_ENUM_NUMBER:
            ''' 非枚举整性变量处理 '''
            value = row[col]  # 获得该列值
            ''' z-score标准化处理 '''
            value_mean = storage[col][2]  # 均值
            value_std = storage[col][4]  # 标准差
            value = (value - value_mean) / value_std
            # print(value)
            ''' 计算灰度图亮度值 '''
            bulb = (value / 1.88) * 127.5 + 127.5
            if bulb < 0:
                bulb = 0
            elif bulb > 255:
                bulb = 255
            gray_list.append(bulb)

        if storage[col][0] == ENUM_NUMBER or storage[col][0] == ENUM_STRING:
            ''' 枚举型变量处理 '''
            value = row[col]  # 获取该列值
            ''' 获得该值在枚举值中的index '''
            value_index = storage[col][-1].index(value)
            ''' 获取该属性枚举值个数 '''
            value_enum = len(storage[col][-1])
            bulb = value_index / (value_enum - 1) * 255
            gray_list.append(bulb)
    ''' 将gray_list转变为array '''
    # print(gray_list)
    # arr = methods.make_matrix(gray_list, round(math.sqrt(element_size)))
    arr = methods.make_matrix(gray_list, 4)
    ''' 判断该用户是否为违约客户 '''
    is_default = methods.is_default(row)
    ''' 生成该客户对应的灰度图图像，命名格式为 counts.jpg '''
    if methods.make_image(arr, pictures_path, row["id"], is_default):
        print("USER {0} MADE SUCCESSFULLY!".format(row["id"]))

''' 全部生成完成后进行重命名 '''
methods.rename()
