from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import os

# CelebA官方数据集下载目录
CelebA_path = "/hy-tmp/Datasets/CelebA/"
# 属性文件
CelebA_Attr_file = CelebA_path + "Anno/list_attr_celeba.txt"
# 输出目录
output_path = CelebA_path + "train/"
# 原图片解压目录
image_path = CelebA_path + "/img_align_celeba"
# 分割属性
ATTR_TYPE = 21  # 男性


def splitImg(attr_type, num=1000):

    trainA_dir = os.path.join(output_path, "A")
    trainB_dir = os.path.join(output_path, "B")
    if not os.path.isdir(trainA_dir):
        os.makedirs(trainA_dir)
    if not os.path.isdir(trainB_dir):
        os.makedirs(trainB_dir)

    not_found_txt = open(os.path.join(output_path, "not_found_img.txt"), "w")

    count_A = 0
    count_B = 0
    count_N = 0

    with open(CelebA_Attr_file, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        Attr_info = Attr_info[2:]
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            filename = info[0]
            # 原图片路径
            filepath_old = os.path.join(image_path, filename)
            if os.path.isfile(filepath_old):
                if int(info[attr_type]) == 1:
                    if count_A >= 1000:
                        continue
                    filepath_new = os.path.join(trainA_dir, filename)
                    shutil.copyfile(filepath_old, filepath_new)
                    count_A += 1
                else:
                    if count_B >= 1000:
                        continue
                    filepath_new = os.path.join(trainB_dir, filename)
                    shutil.copyfile(filepath_old, filepath_new)
                    count_B += 1
#                 print("%d: success for copy %s -> %s" % (index, info[attr_type], filepath_new))
            else:
                print("%d: not found %s\n" % (index, filepath_old))
                not_found_txt.write(line)
                count_N += 1

    not_found_txt.close()

    print("TrainA have %d images!" % count_A)
    print("TrainB have %d images!" % count_B)
    print("Not found %d images!" % count_N)

    
if __name__ == "__main__":
    splitImg(ATTR_TYPE)
