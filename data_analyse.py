import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "D:\\Kaggle\\DataSets\\accepted_2007_to_2018q4.csv\\accepted_2007_to_2018Q4.csv"
test_data_path = "D:\\Kaggle\\DataSets\\accepted_2007_to_2018q4.csv\\test.csv"

data = pd.read_csv(data_path)
mean_num = data["loan_amnt"].mean()
std_num = data["loan_amnt"].std()
print(mean_num, std_num)
# 区间1000分割
place_piece = 5000
x = []
y = []
num_dict = {}
for num in data["loan_amnt"]:
    if pd.isnull(num):
        continue
    place = int(num/place_piece)
    if place in num_dict:
        num_dict[place] += 1
    else:
        num_dict[place] = 1
for key in num_dict.keys():
    x.append(key)
    y.append(num_dict[key])
for i in range(len(x)):
    for j in range(i, len(x)):
        if x[i] > x[j]:
            temp = x[i]
            x[i] = x[j]
            x[j] = temp
            temp = y[i]
            y[i] = y[j]
            y[j] = temp
plt.plot(x, y)
plt.show()


