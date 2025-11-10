#文件修改
import os

def question3():
    filename = "test.txt"
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"'{filename}' doesn't exist")
        print("run test12 first")
        return
    
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        print(f"read successfully, string number is{len(original_content)} ")
        
    except Exception as e:
        print(f"文件读取失败: {e}")
        return
    
    # 2. 在开头和结尾添加"python"
    print("正在修改文件内容...")
    
    # 使用文件操作方法：在开头和结尾添加"python"
    new_content = "python\n" + original_content + "\npython"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("文件修改成功！")
       
        
       
    except Exception as e:
        print(f"failed{e}")

# 独立运行第三题
if __name__ == "__main__":
    question3()
