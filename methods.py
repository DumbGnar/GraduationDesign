import settings as st
import numpy as np


def is_default(row):
    """ 判断是否违约，违约时返回True """
    return row["loan_status"].startswith("Does not") or \
           row["loan_status"].startswith("Late") or \
           row["loan_status"].startswith("In Grace")


def value_is_default(string_value):
    """ 判断某个值是否属于违约范围，违约时返回True """
    return string_value.startswith("Does not") or \
           string_value.startswith("Late") or \
           string_value.startswith("In Grace")


def get_type_str(type_enum):
    """ 根据type值来返回一个str，表示属性类别 """
    if type_enum == st.ENUM_STRING:
        return "枚举字符型"
    elif type_enum == st.ENUM_NUMBER:
        return "枚举数型"
    elif type_enum == st.NOT_ENUM_NUMBER:
        return "非枚举数型"
    elif type_enum == st.NOT_ENUM_STRING:
        return "非枚举字符型"
    return "未知类型"


def statics_save(cols_iterable=None, storage_dict=None, filename="generated"):
    """ 用于形成存储 """
    ''' 将统计文件字典进行存储 '''
    np.save(st.base + filename + ".npy", storage_dict)
    ''' 形成可读的文件 '''
    lines = []
    with open(st.base + filename + ".txt", mode="w") as f:
        for col in cols_iterable:
            to_write = ""
            to_write += "列名: " + col + "\n"
            to_write += "        属性类型: " + str(get_type_str(storage_dict[col][0])) + "\n"
            if storage_dict[col][1]:
                to_write += "        信息价值: " + str(round(storage_dict[col][1], 3)) + "\n"
            else:
                to_write += "        信息价值: " + "None" + "\n"

            if storage_dict[col][2]:
                to_write += "        均值: " + str(round(storage_dict[col][2], 3)) + "\n"
            else:
                to_write += "        均值: " + "None" + "\n"

            if storage_dict[col][3]:
                to_write += "        众数: " + str(storage_dict[col][3]) + "\n"
            else:
                to_write += "        众数: " + "None" + "\n"

            if storage_dict[col][4]:
                to_write += "        标准差: " + str(round(storage_dict[col][4], 3)) + "\n"
            else:
                to_write += "        标准差: " + "None" + "\n"
            to_write += "\n"
            lines.append(to_write)
        f.writelines(lines)
        f.flush()


def statics_load(filename="generated"):
    """ 加载所保存的统计数据字典 """
    data = np.load(st.base + filename + ".npy")
    return data
