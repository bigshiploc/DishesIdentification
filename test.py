#coding=utf-8
import os
# 列出当前目录下所有的文件
path = input('Input dir : ')
files = os.listdir(path)
for filename in files:
    portion = os.path.splitext(filename)
    # 如果后缀是.txt
    if portion[1] == ".jpg":
        # 重新组合文件名和后缀名
        newname = portion[0] + "te.jpg"
        # print newname
        os.rename(path+"/"+filename,path+"/"+newname)