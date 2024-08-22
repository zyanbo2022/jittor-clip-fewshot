import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import jittor as jt
from jittor import transform as jt_transform
from clip.moco import load_moco

from datasets.utils import DatasetWrapper

from datasets.TrainSet import *
from run_utils import *
import clip
from utils import *
from utils import Resize,_convert_image_to_rgb
import json
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from jittor.transform import Compose, ImageNormalize

def extract_text_feature403(cache_dir, classnames, clip_model, template):
    jt.flags.use_cuda = 1
    with jt.no_grad():
        clip_weights = []
        for classname in classnames:

            texts = [classname]
            texts_token = clip.tokenize(texts, truncate=True)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
            del classname
            del class_embedding
            del class_embeddings
            jt.gc()

        clip_weights = jt.stack(clip_weights, dim=1)
    jt.save(clip_weights, cache_dir + "/text_weights403.pkl")
    return
    
def extract_text_feature374(cache_dir, classnames, clip_model, template):
    jt.flags.use_cuda = 1
    with jt.no_grad():
        clip_weights = []
        for classname in classnames:

            texts = [classname]
            texts_token = clip.tokenize(texts, truncate=True)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
            del classname
            del class_embedding
            del class_embeddings
            jt.gc()

        clip_weights = jt.stack(clip_weights, dim=1)
    jt.save(clip_weights, cache_dir + "/text_weights374.pkl")
    return


def extract_multi_scale_feature1(cache_dir, split, model, loader, scale,tfm_norm,name):
    jt.flags.use_cuda = 1
    features, labels = [], []
    with jt.no_grad():
        for crop_idx in range(1):
            features_this = []
            for i, (images, target) in enumerate(tqdm(loader)):
                # images = images.cuda()
                if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                    image_features = model.encode_image(tfm_norm(images))  # for clip model
                else:
                    image_features = model(tfm_norm(images))
                # image_features = clip_model.encode_image(images)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                features_this.append(image_features)
                if crop_idx == 0:
                    target =  jt.array(target)
                    labels.append(target)
                del images
                del target
                del image_features
                jt.gc()  # 强制释放显存
                
            features.append(jt.cat(features_this, dim=0))
    features, labels = jt.stack(features, dim=0), jt.cat(labels)
    print("下面输出维度", features.shape, labels.shape,type(features),type(labels))
    jt.save(features, cache_dir + "/" + name +"_" + split + "_f"+ "_" + str(scale) + ".pkl")
    label_path = cache_dir + "/" + name+"_" + split + "_l.pkl"
    if not os.path.exists(label_path):
        jt.save(labels, label_path)
    return

def extract_multi_scale_feature(cache_dir, split, model, loader, scale,tfm_norm,name):
    jt.flags.use_cuda = 1
    features, labels = [], []
    with jt.no_grad():
        for crop_idx in range(10):
            features_this = []
            for i, (images, target) in enumerate(tqdm(loader)):
                # images = images.cuda()
                if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                    image_features = model.encode_image(tfm_norm(images))  # for clip model
                else:
                    image_features = model(tfm_norm(images))
                # image_features = clip_model.encode_image(images)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                features_this.append(image_features)
                if crop_idx == 0:
                    target =  jt.array(target)
                    labels.append(target)
                del images
                del target
                del image_features
                jt.gc()  # 强制释放显存
                
            features.append(jt.cat(features_this, dim=0))
    features, labels = jt.stack(features, dim=0), jt.cat(labels)
    print("下面输出维度", features.shape, labels.shape,type(features),type(labels))
    jt.save(features, cache_dir + "/" + name +"_" + split + "_f"+ "_" + str(scale) + ".pkl")
    
    label_path = "./caches/trainset" + "/" + name+"_" + split + "_l.pkl"
    if not os.path.exists(label_path):
        jt.save(labels, label_path)
    return




if __name__ == '__main__':
    
    tfm_clip = Compose([ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    tfm_aux = Compose([ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    clip_model, preprocess = clip.load('ViT-B-32.pkl')
    clip_model.eval()
    argslora = get_arguments()
    list_lora_layers = apply_lora(argslora, clip_model)
    load_lora(argslora, list_lora_layers)
    clip_model.eval()

    # 定义缓存目录
    cache_dir = os.path.join('./caches', 'trainset0')
    os.makedirs(cache_dir, exist_ok=True)
    
    
    aux_model, feat_dim = load_moco("r-50-1000ep.pkl")

    aux_model.eval()
            

    # 设置随机种子
    random.seed(1)
    # numpy.random.seed(1)
    jt.set_global_seed(1)
    jt.seed(1)
    jt.set_seed(1)

    
    # 加载数据集
    dataset = TrainSet_double()
    
    classnames = open('prompt/b.txt').read().splitlines()
    extract_text_feature403(cache_dir, classnames , clip_model, dataset.template)

    classnames = open('prompt/a.txt').read().splitlines()
    extract_text_feature374(cache_dir, classnames , clip_model, dataset.template)
    
    # 循环处理不同的尺度
    for this_scale in range(1, 11):
        print(f"\nProcessing : {this_scale}")
        print(f"\nProcessing scale: {this_scale * 0.1}")


        if this_scale==10:
            test_transform = jt_transform.Compose([

                Resize(224, mode=Image.BICUBIC),
                jt_transform.CenterCrop(224),
                _convert_image_to_rgb,
                jt_transform.ToTensor(),

            ])
        else:
            test_transform = jt_transform.Compose([
                jt_transform.RandomCropAndResize(size=224, scale=(this_scale * 0.1, this_scale * 0.1), interpolation=Image.BICUBIC),
                jt_transform.CenterCrop(224),
                _convert_image_to_rgb,
                jt_transform.ToTensor(),

            ])



        val_loader = DatasetWrapper(data_source=dataset.val, batch_size=128, is_train=False, tfm=test_transform, shuffle=False)
        test_loader = DatasetWrapper(data_source=dataset.test, batch_size=128, is_train=False, tfm=test_transform, shuffle=False)

    
        # 提取clip多尺度特征
        print("\nLoading clip val visual features and labels.")
        if this_scale==10:
            extract_multi_scale_feature1(cache_dir, "val", clip_model, val_loader, this_scale,tfm_clip,"clip")
        else:
            extract_multi_scale_feature(cache_dir, "val", clip_model, val_loader, this_scale,tfm_clip,"clip")
            
        print("\nLoading clip test visual features and labels.")
        if this_scale==10:
            extract_multi_scale_feature1(cache_dir, "test", clip_model, test_loader, this_scale,tfm_clip,"clip")
        else:
            extract_multi_scale_feature(cache_dir, "test", clip_model, test_loader, this_scale,tfm_clip,"clip")

       
    