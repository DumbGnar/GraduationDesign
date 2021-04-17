""" 全局参数设置 """
base = "D:\\Kaggle\\DataSets\\accepted_2007_to_2018q4.csv\\"
data_name = "accepted_2007_to_2018Q4.csv"
test_data_name = "test.csv"
data_path = base + data_name
test_data_path = base + test_data_name
test_mode = True  # 为True时使用小数据集
delete_rate = 0.50  # 数据预处理接段如果超过该值则会剔除该行对应数据
enum_upper = 12 * 12  # 当一列中出现的值种类少于该值时认定为枚举类
ENUM_NUMBER = 0  # 枚举型整数数据
ENUM_STRING = 1  # 枚举型字符数据
NOT_ENUM_NUMBER = 2  # 非枚举型整型数据
NOT_ENUM_STRING = 3  # 非枚举型字符数据
UNKNOWN = 4  # 未知情况类型
group_max_num = 10  # 对于非枚举型变量分组中最多有几组
IV_critical = 0.30  # 对于IV值低于IV_critical的属性进行剔除
