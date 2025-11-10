#文件复制
'''
import random
import string

def second_question():
    try:
        line_count = int(input("请输入要生成的行数: "))
    except ValueError:
        print("输入错误，请输入一个数字！")
        return
    
    source_file = "test.txt"
    target_file = "copy_test.txt"
    
    # 第一部分：创建原始文件
    print(f"1. 正在创建包含 {line_count} 行随机字符的文件...")
    
    # ASCII标准字符：包括字母、数字、标点符号等可打印字符
    # string.printable 包含了所有可打印的ASCII字符
    ascii_characters = string.printable
    
    with open(source_file, 'w', encoding='utf-8') as file:
        for i in range(line_count):
            # 每行生成随机长度的内容（10-50个字符）
            line_length = random.randint(10, 50)
            
            # 生成一行随机字符
            line_content = ''.join(random.choices(ascii_characters, k=line_length))
            
            # 写入文件，每行末尾加换行符
            file.write(line_content + '\n')
    
    print(f"✅ 原始文件 '{source_file}' 创建成功！")
    
    # 第二部分：复制文件
    print("2. 正在复制文件...")
    
    try:
        # 打开原始文件用于读取，复制文件用于写入
        with open(source_file, 'r', encoding='utf-8') as src:
            with open(target_file, 'w', encoding='utf-8') as dst:
                # 逐行读取原始文件并写入目标文件
                for line in src:
                    dst.write(line)
        
        print(f"✅ 文件复制成功：'{source_file}' -> '{target_file}'")
        
    except Exception as e:
        print(f"❌ 文件复制失败：{e}")

# 运行第二题
if __name__=="main":
    second_question
'''
# 第二题：文件复制
import random
import string

def question2():
    """
    第二题：文件复制
    功能：创建包含随机ASCII字符的文件，并复制该文件
    """
    line_count = int(input("请输入要生成的行数: "))
        
    
    source_file = "test.txt"
    target_file = "copy_test.txt"
    
    # 1. 创建原始文件（包含ASCII标准字符）
    print(f"1. 正在创建包含 {line_count} 行ASCII字符的文件...")
    
    # ASCII标准字符：所有可打印字符（字母、数字、标点、空格等）
    ascii_chars = string.printable
    
    with open(source_file, 'w', encoding='utf-8') as obj_file:
        for i in range(line_count):
            # 每行生成5-20个随机字符
            char_count = random.randint(5, 20)
            line_content = ''.join(random.choices(ascii_chars, k=char_count))
            obj_file.write(line_content + '\n')
    
    print(f"original file'{source_file}' was created")
    
    # 2. 复制文件
    print("copy file now")
    
    #try:
    with open(source_file, 'r', encoding='utf-8') as src:
         with open(target_file, 'w', encoding='utf-8') as dst:
                # 逐行复制内容
            content = src.read()
            dst.write(content)
        
    print(f"copy successfully")
        
        # 显示文件信息
        
# 独立运行第二题
if __name__ == "__main__":
    question2()

