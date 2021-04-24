import methods

''' 形成一个长度为90的list '''
test_list = []
for i in range(90):
    test_list.append(i + 1)
arr = methods.make_matrix(test_list, 2)
print(arr)
