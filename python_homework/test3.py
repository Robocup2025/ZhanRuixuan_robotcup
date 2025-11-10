# 此为第1页第3题
a=[0]*20
a[1]=1
for i in range(2,20):
    a[i]=a[i-1]+a[i-2]
for i in range(0,20):
    print(a[i],end=' ')
    