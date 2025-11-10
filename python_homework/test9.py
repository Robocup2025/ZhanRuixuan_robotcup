# 创建顺序文件
import os,random,string
def creat_file():
    folder_path='img'
    if not os.path.exists(folder_path):#os.path.exists()用于判断路径是否存在
        os.makedirs(folder_path)#创建文件夹
        print(f'creat folder{folder_path}')
    else:
        print(f'{folder_path}exist')
    existing_file=set()#集合set用于存储已使用的文件名
    count=0
    print('creat 100 file')
    for i in range(100):
        un=False#注意大写
        filename=''
        while not un:
            random_chars=''.join(random.choices(string.ascii_uppercase+string.digits,k=4))
            # 这是生成4个随机字符（大写字母+数字）
            # random.choices():从序列中随机选取元素
            # string.ascii_uppercase：所有大写字母
            # string.digits所有数字
            # join为将多个字符串连接在一起，用法：连接符.join(字符串列表)
            snum=str(i+1).zfill(3)#zfill()为用0填充到指定长度
            filename=f"{random_chars}{snum}.png"
            if filename not in existing_file:
                existing_file.add(filename)
                un=True
        f_path=os.path.join(folder_path,filename)
        # 上行创建了完整文件路径，os.path.join是智能拼接文件路径
        with open(f_path,'w')as file_obj:
            pass
if __name__=="__main__":
    creat_file()

