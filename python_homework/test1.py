# 此为第一题
cout=0
for i in range(1,5):
    for j in range(1,5):
        for k in range(1,5):
            if i!=j and j!=k and i!=k:
                a=100*i+10*j+k
                print(a,end=' ')
                cout=cout+1
print(f"\n共{cout}个")



