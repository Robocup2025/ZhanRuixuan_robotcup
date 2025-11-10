#文件对比

import os

def question4():
    """
    第四题：文件对比
    功能：比较两个文件的每一行内容，输出不同的行号
    """
    print("=== 第四题：文件对比 ===")
    
    file1 = "test.txt"
    file2 = "copy_test.txt"
    
    # 检查文件是否存在
    if not os.path.exists(file1):
        print("run test12.py")
        return
    
    if not os.path.exists(file2):
        print("run test12.py")
        return
    
    print("now compare the two test")
    
    different_lines = []  # 存储不同行的行号
    
    try:
        # 同时打开两个文件进行逐行比较
        with open(file1, 'r', encoding='utf-8') as f1, \
             open(file2, 'r', encoding='utf-8') as f2:
            
            line_num = 1
            different_count = 0
            
            # 逐行比较
            while True:
                line1 = f1.readline()#每次读取文件的一行
                line2 = f2.readline()
                
                # 如果两个文件都读完了
                if not line1 and not line2:
                    break
                
                # 去除换行符后比较内容
                content1 = line1.rstrip('\n') if line1 else None
                content2 = line2.rstrip('\n') if line2 else None
                
                if content1 != content2:
                    different_lines.append(line_num)
                    different_count += 1
                    print(f"第{line_num}行不同:")
                    print(f"  {file1}: {content1}")
                    print(f"  {file2}: {content2}")
                    print("-" * 40)
                
                line_num += 1
            
            # 输出比较结果
            if different_count == 0:
                print("✅ 两个文件内容完全相同！")
            else:
                print(f"\n📊 比较结果:")
                print(f"总行数检查到: {line_num-1} 行")
                print(f"不同行数: {different_count} 行")
                print(f"不同行号: {different_lines}")
                
    except Exception as e:
        print(f"文件比较失败: {e}")

# 独立运行第四题
if __name__ == "__main__":
    question4()
