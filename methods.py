import settings as st
import numpy as np
import os
from PIL import Image
import random
import gzip
import struct


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


def build():
    """ 生成该文件对应的灰度图文件夹，并返回建立路径，返回地路径中末尾已经带有\\ """
    ''' 生成存储基础路径 '''
    data_path = st.base + "data_pictures\\"
    if os.path.exists(data_path):
        os.removedirs(data_path)
    os.mkdir(data_path)
    ''' 建立训练数据集和测试数据集 '''
    test_data_path = data_path + "testing\\"
    train_data_path = data_path + "training\\"
    os.mkdir(test_data_path)
    os.mkdir(train_data_path)
    ''' 建立违约客户(1)和正常客户(0)的测试集图像文件夹 '''
    default_in_test_path = test_data_path + "1\\"
    not_default_test_path = test_data_path + "0\\"
    os.mkdir(not_default_test_path)
    os.mkdir(default_in_test_path)
    ''' 建立违约客户(1)和正常客户(0)的训练集图像文件夹 '''
    default_in_train_path = train_data_path + "1\\"
    not_default_train_path = train_data_path + "0\\"
    os.mkdir(not_default_train_path)
    os.mkdir(default_in_train_path)
    ''' 返回基路径 '''
    return data_path


def make_matrix(gray_list, size):
    """ 该方法返回由list生成的32*32矩阵，矩阵中每一个属性占size*size个单元格 """
    res = np.zeros((32, 32))
    ''' 计算各行各列最多放几个元素 '''
    max_size = 32 / size
    ''' 对list进行长度扩充 '''
    while len(gray_list) < max_size**2:
        gray_list.append(255)
    # print(len(gray_list))
    ''' 进行数据填充 '''
    for index in range(len(gray_list)):
        ''' 行和列的计算 '''
        row = int(index / max_size)
        col = index % int(max_size)
        # print(row, col)
        ''' 根据计算的行和列求出具体的截取范围 '''
        start_row = row * size
        end_row = (row + 1) * size
        start_col = col * size
        end_col = (col + 1) * size
        # print(start_row, end_row, "||", start_col, end_col)
        ''' 进行赋值 '''
        res[start_row:end_row, start_col:end_col] = gray_list[index]
    return res


def make_image(gray_array, pictures_path, uid, is_default):
    """ 该方法根据array生成灰度图，命名格式为 {uid}.jpg，方法返回生成成功与否 """
    if gray_array is None:
        return False
    ''' 根据settings中的比例选择来将图像放入训练集还是测试集 '''
    is_train = None
    if is_default:
        ''' 违约客户 '''
        is_train = (random.random() <= st.default_training_rate)
    else:
        ''' 非违约客户 '''
        is_train = (random.random() <= st.not_default_training_rate)
    if is_train:
        ''' 训练集数据 '''
        pictures_path += "training\\"
    else:
        ''' 测试集数据 '''
        pictures_path += "testing\\"
    if is_default:
        pictures_path += "1\\"
    else:
        pictures_path += "0\\"
    path = pictures_path + str(uid) + ".jpg"
    ''' 如果之前已经建立该用户的灰度图，则清空重建 '''
    if os.path.exists(path):
        os.remove(path)
    img = Image.fromarray(np.uint8(gray_array))
    img.save(path)
    return True


def _read(image, label):
    """读取数据的函数,先读取标签，再读取图片。解压标签包"""
    with gzip.open(label) as flbl:
        ''' 采用Big Endian的方式读取两个int类型的数据，且参考MNIST官方格式介绍，magic即为magic number (MSB first) 
        用于表示文件格式，num即为文件夹内包含的数据的数量'''
        magic, num = struct.unpack(">II", flbl.read(8))
        '''将标签包中的每一个二进制数据转化成其对应的十进制数据，且转换后的数据格式为int8（-128 to 127）格式，返回一个数组'''
        label = np.frombuffer(flbl.read(), dtype=np.int8)
    '''以只读形式解压图像包'''
    with gzip.open(image, 'rb') as fimg:
        '''采用Big Endian的方式读取四个int类型数据，且参考MNIST官方格式介绍，magic和num上同，rows和cols即表示图片的行数和列数'''
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        '''将图片包中的二进制数据读取后转换成无符号的int8格式的数组，并且以标签总个数，行数，列数重塑成一个新的多维数组'''
        image = np.frombuffer(fimg.read(), dtype=np.uint8)
        image = image.reshape(len(label), rows, cols)
    return image, label


def get_data():
    """
        调用方法
        train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig = get_data()
    """
    train_img, train_label = _read(
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz')

    test_img, test_label = _read(
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz')
    return [train_img, train_label, test_img, test_label]
