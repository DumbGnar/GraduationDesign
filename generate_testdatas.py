import pandas as pd
import settings as st

''' 本程序用于生成编写程序时的小规格数据集 '''
data_path = st.data_path
test_data_path = st.test_data_path
''' 读入源数据,并保留前50行 '''
data = pd.read_csv(data_path, nrows=5000)
''' 保存到测试数据路径 '''
data.to_csv(test_data_path, index=None)
