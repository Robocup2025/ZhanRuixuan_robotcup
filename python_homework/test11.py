# 文件输入
import os,random,string,statistics

def creat_folder():
    foldername='zhanruixuan.txt'
    k=[]
    if not os.path.exists(foldername):
        with open(foldername,'w',encoding='utf-8')as file_obj:
            for i in range(10):
                r1=random.randint(1,100)
                r2=random.randint(1,100)
                r3=random.randint(1,100)
                k.append([r1,r2,r3])
                file_obj.write(f'{r1},{r2},{r3}\n')#注意是file_obj不是foldername
            print('open successfully')
    else:
        # 若已有文件则读取
        with open(foldername,'r',encoding='utf_8')as file_obj:
            lines=file_obj.readlines()#读取行数
            for line in lines:
                numbers=line.strip().split(',')#line.strip()为去除字符串两端空白字符，.split为以，为分隔符将字符串分割为多部分
                if len(numbers)==3:
                    k.append([int(numbers[0]),int(numbers[1]),int(numbers[2])])
            print('read successfully')
    max_value=0
    for i in range(10):
        if max_value<=k[i][1]:
            max_value=k[i][1]
    min_v=100
    for i in range(10):
        if min_v>=k[i][1]:
            min_v=k[i][1]
    sum_v=0
    for i in range(10):
        sum_v=sum_v+k[i][1]
    sum_v=sum_v/10
    # 以下用函数找中位数
    m=[k[0][1]]
    for i in range(1,10):
        m.append(k[i][1])
    median_v=statistics.median(m)
    print(f"最大值{max_value}最小值{min_v}平均值{sum_v}中位数{median_v}")


if __name__ =="__main__":
    creat_folder()




