# 此为第1页第2题
a=int(input("num1"))
b=int(input('num2'))
c=int(input('num3'))
if a>=b:
    x=a
    a=b
    b=x
if b>=c:
    y=b
    b=c
    c=y
print(f"排序结果{a}{b}{c}")