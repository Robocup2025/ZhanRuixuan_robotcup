# 随机数文件
import random
file=open('data.txt','w',encoding='utf-8')#学长其实encoding可以不写，我写只是为了保险
for i in range(100000):
    num=random.randint(1,100)
    file.write(str(num)+'\n')
file.close()