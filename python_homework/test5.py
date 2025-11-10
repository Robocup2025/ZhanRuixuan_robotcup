#回文数
a=int(input('输入数'))
b=str(a)
c=b[::-1]
d=int(c)
if a==d:
    print('yes')
else:
    print('no')