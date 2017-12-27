#coding=utf-8
import os
# 列出当前目录下所有的文件
files = os.listdir("./data/test/Yu-ShiangShreddedPork")
for filename in files:
    portion = os.path.splitext(filename)
    # 如果后缀是.txt
    if portion[1] == ".jpeg":
        # 重新组合文件名和后缀名
        newname = portion[0] + ".jpg"
        # print newname
        os.rename("./data/test/Yu-ShiangShreddedPork/"+filename,"./data/test/Yu-ShiangShreddedPork/"+newname)