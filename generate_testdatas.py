import pandas as pd
import grayImage as gi
''' 本程序用于生成编写程序时的小规格数据集 '''
''' 读入源数据,并保留前50行 '''
data = pd.read_csv(gi.data_path, nrows=50)
''' 保存到测试数据路径 '''
data.to_csv(gi.test_data_path)
