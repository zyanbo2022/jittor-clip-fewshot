#!/bin/bash

# 训练集测试集的图片根目录在此修改,例如可替换为“/root/pic_root”,确保训练集测试集在此文件夹下。
# 或者直接将数据集放在caches目录下

root_path="caches"

# 使用 sed 进行替换，并确保替换后的路径带双引号
sed -i "s|^root_path: .*|root_path: \"${root_path}\"|" default.yaml

python delete_file.py
# 训练lora
python lora_main.py
# 多尺度特征提取、筛选、融合
python extract_features.py
python feat-extract.py
python feat-selection.py
python feat-merge.py
#训练分类器生成少量伪标签
python train-double.py
#训练最终分类器
python train-amu.py
