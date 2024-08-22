

# Jittor “代码写的都对”队伍  开放域少样本视觉分类赛题  

## 简介

本项目包含了第四届计图挑战赛计图 - 开放域少样本视觉分类赛题的代码实现。
- 1.采用了 clip-lora方法对clip模型微调处理。
- 2.从多个尺度（0.1到1.0）对图像随机切分，并筛选特征，最后融合特征。
- 3.结合tip-adapter和AMU-Tuning训练分类器，生成生成的伪标签数据。
- 4.对生成的伪标签数据（每个类别16张）再加入训练集，重复训练分类器。
- 5.最终在B榜测试集取得了最高75.96%的效果。

## 安装 

本项目在24G的3090显卡上运行，训练时间约为2小时。

#### 运行环境
- ubuntu 20.04
- python == 3.8
- jittor == 1.3.8

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

#### 预训练模型下载并转换为pkl

本项目需要两个预训练模型：
- 1.clip的ViT-B-32模型的jittor版本，需要放到代码根目录下。ViT-B-32.pkl 下载链接https://github.com/uyzhang/JCLIP/releases/download/%E6%9D%83%E9%87%8D/ViT-B-32.pkl。
- 2.mocov3的预训练模型，预训练模型模型下载地址为 https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar，
下载到根目录，需要通过转换代码zh.py，将r-50-1000ep.pth.tar转换为jittor版本，r-50-1000ep.pkl。
- 3.r-50-1000ep.pkl模型参数量为23.56M,经过Lora微调的ViT-B-32.pkl模型参数量为151.47M，两个线性分类器的参数量为5.4M。
## 数据集下载解压


将数据下载并解压到 `<root>/caches` ，也可在bash中修改路径：

- 1.训练集caches/TrainSet
- 2.测试集集caches/TestSetA


## 训练

训练可运行以下命令：
```
bash train.sh
```

## 推理

```
python test.py
```
测试集结果在训练完成自动生成，存放在result/result.txt。

qq1447739028