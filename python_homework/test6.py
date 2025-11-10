# problem：在删除奇数时列表长度会变，会跳过某个元素,ide的值会超过列表长度，以下为修改后，我创建了一个新数组
list = list(range(1000))
new_list = []
for num in list:
    if num % 2 == 0:
        new_list.append(num)
print(new_list)#学长若觉得无需输出可以删去
