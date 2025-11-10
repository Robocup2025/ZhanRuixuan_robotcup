#文件改名
import os,random

def rename_files_randomly():
    FOLDER_NAME='img'
    RENAME_COUNT=50
    NEW_EXTENSION=".jpg"
    if not os.path.exists(FOLDER_NAME):
        print("run test9.py first")
        return False
    all_files=os.listdir(FOLDER_NAME)
    #上行函数为列出目录中所有文件和文件夹
    file_list=[filename for filename in all_files
               if os.path.isfile(os.path.join(FOLDER_NAME,filename))
               ]
    # 上行为过滤出真正的文件，排除文件夹，提取文件名，以下为更通俗代码
    '''
    file_list=[]
    for filename in all_files:
        full_path=os.path.join(FOLDER_NAME,filename)
        if os.path.isfle(full_path):#检查是否为文件
            file_list.append(filename)#这是添加到列表的函数
    
    '''
    if len(file_list)<RENAME_COUNT:
        print('wrong')
        return False
    files_to_rename=random.sample(file_list,RENAME_COUNT)
    rename_success=0
    for old_filename in files_to_rename:
        old_filepath=os.path.join(FOLDER_NAME,old_filename)
        # 以下是构建新文件名，os.path.splitext()是分割文件名和拓展名
        name_part,ext_part=os.path.splitext(old_filename)
        new_filename=name_part+NEW_EXTENSION
        new_filepath=os.path.join(FOLDER_NAME,new_filename)

        # 以下为重命名文件
        # os.rename():重命名文件或目录
        os.rename(old_filepath,new_filepath)

    print('job done')
    return True
if __name__=="__main__":
    rename_files_randomly()