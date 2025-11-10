# 最后剩下谁
man=list(range(1,234))
index=0
while len(man)>1:
    index=(index+2)%len(man)
    man.pop(index)
print(man)