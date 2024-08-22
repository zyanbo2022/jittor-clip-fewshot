import os
import shutil

# 定义要检查和删除的文件路径和文件夹路径
file_path = "caches/temp_labels.txt"
# dir_path = "caches/trainset"

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"文件 {file_path} 已删除")
    else:
        print(f"文件 {file_path} 不存在")

def delete_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"文件夹 {dir_path} 已删除")
    else:
        print(f"文件夹 {dir_path} 不存在")

if __name__ == "__main__":
    delete_file(file_path)
    # delete_dir(dir_path)