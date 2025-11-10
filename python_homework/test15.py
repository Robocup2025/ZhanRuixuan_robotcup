# 第五题：文件批量创建 - 独立运行版本
import os
import random
import string

def question5():
    try:
        file_count = int(input("请输入要创建的文件数量: "))
        line_count = int(input("请输入每个文件的行数: "))
        
        if file_count <= 0 or line_count <= 0:
            print("文件数量和行数必须大于0")
            return
            
    except ValueError:
        print("输入错误，请输入正整数！")
        return
    
    directory = "test"
    
    # 1. 创建test目录（如果已存在则清空）
    if os.path.exists(directory):
        # 删除目录及其所有内容
        import shutil
        shutil.rmtree(directory)
        print(f"目录 '{directory}' 已存在，已清空")
    
    os.makedirs(directory)
    print(f"目录 '{directory}' 创建成功")
    
    # 2. 创建指定数量的文件
    print(f"2. 正在创建 {file_count} 个文件，每个文件 {line_count} 行...")
    
    original_files = []
    
    for i in range(1, file_count + 1):
        filename = f"file{i}.txt"
        filepath = os.path.join(directory, filename)
        original_files.append(filepath)
        
        # 创建文件并写入随机内容
        with open(filepath, 'w', encoding='utf-8') as f:
            for j in range(line_count):
                # 每行生成3-10个随机字符（字母和数字）
                char_count = random.randint(3, 10)
                line_content = ''.join(random.choices(
                    string.ascii_letters + string.digits, k=char_count
                ))
                f.write(line_content + '\n')
        
        print(f"   创建文件: {filename}")
    
    print("✅ 所有原始文件创建完成")
    
    # 3. 修改文件名和文件内容（添加-python）
    print("3. 正在修改文件名和文件内容...")
    
    modified_count = 0
    
    for filepath in original_files:
        if os.path.exists(filepath):
            # 读取原文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 在每个行末尾添加"-python"
            new_lines = []
            for line in lines:
                cleaned_line = line.rstrip('\n')  # 去除换行符
                new_lines.append(cleaned_line + "-python\n")  # 添加-python和换行符
            
            # 生成新文件名（添加-python）
            dir_name = os.path.dirname(filepath)
            base_name = os.path.basename(filepath)
            name, ext = os.path.splitext(base_name)
            new_filename = f"{name}-python{ext}"
            new_filepath = os.path.join(dir_name, new_filename)
            
            # 写入新文件
            with open(new_filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            # 删除原文件
            os.remove(filepath)
            
            modified_count += 1
            print(f"   修改完成: {os.path.basename(filepath)} -> {new_filename}")
    
    print(f"✅ 文件修改完成！共处理了 {modified_count} 个文件")
    print("   所有文件名和每行内容都已添加'-python'后缀")
    
    # 显示修改结果示例
    
# 独立运行第五题
if __name__ == "__main__":
    question5()
