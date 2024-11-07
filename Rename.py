import os

# 设置文件夹路径
folder_path = 'result/NUAA-Swin Transformer/visulization_result'

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # 构建旧文件名和新文件名
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, f'NUAA-Swin Transformer-{filename}')

        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')

print("重命名完成")



# # 设置要遍历的根目录
# root_folder = 'result/NUAA-09'
#
# # 遍历根目录及其子文件夹
# for subdir, dirs, files in os.walk(root_folder):
#     for file in files:
#         if '_Pred' not in file and file.endswith('.png'):
#             # 构建文件路径
#             file_path = os.path.join(subdir, file)
#
#             # 删除文件
#             os.remove(file_path)
#             print(f'Deleted: {file_path}')
#
# print("删除完成")
