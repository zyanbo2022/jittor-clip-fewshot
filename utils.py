import logging
import datetime
from PIL import Image
import jittor as jt
from jittor.dataset.dataset import Dataset
from jittor import transform as jt_transform
from tqdm import tqdm
from jittor import nn

import clip
import cv2
import numpy as np
import os
import random
import os.path as osp
from collections import defaultdict
import gdown


class Resize:
    def __init__(self, size, mode=Image.BILINEAR):
        if isinstance(size, int):
            self.size = size
        else:
            self.size = _setup_size(
                size,
                error_msg="If size is a sequence, it should have 2 values")
        self.mode = mode

    def __call__(self, img: Image.Image):
        if not isinstance(img, Image.Image):
            img = to_pil_image(img)
        if isinstance(self.size, int):
            w, h = img.size

            short, long = (w, h) if w <= h else (h, w)
            if short == self.size:
                return img

            new_short, new_long = self.size, int(self.size * long / short)
            new_w, new_h = (new_short, new_long) if w <= h else (new_long,
                                                                 new_short)
            size = (new_h, new_w)
        return jt_transform.resize(img, size, self.mode)

def _convert_image_to_rgb(image):
    return image.convert("RGB")




tfm_train_base = jt_transform.Compose([
    jt_transform.RandomCropAndResize(size=224, scale=(0.5, 1), interpolation=Image.BICUBIC),
    jt_transform.RandomHorizontalFlip(p=0.5),
    jt_transform.CenterCrop(224),
    jt_transform.ToTensor()
])



tfm_test_base = jt_transform.Compose([
    Resize(224, mode=Image.BICUBIC),
    jt_transform.CenterCrop(224),
    _convert_image_to_rgb,
    jt_transform.ToTensor()
])





def cls_acc(output, target):


    
    l_argmax = output.argmax(dim=-1)
    argmax_result = l_argmax[0]


    correct_predictions = (argmax_result == target).sum()
    acc = 100 * correct_predictions/ target.shape[0]

    return acc

def clip_classifier(classnames,  clip_model):
    clip_weights = []
    jt.flags.use_cuda = 1
    for classname in classnames:
        texts = [classname]
        texts = clip.tokenize(texts)
        texts =  jt.array(texts)
        class_embeddings = clip_model.encode_text(texts)

        
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        clip_weights.append(class_embedding)

        del classname
        del class_embeddings
        jt.gc()
    clip_weights = jt.stack(clip_weights, dim=1)
    return clip_weights


def clip_classifier_s(classname_files, clip_model):
    all_clip_weights = []
    jt.flags.use_cuda = 1

    for filename in classname_files:
        classnames = open(filename).read().splitlines()
        clip_weights = []
        for classname in classnames:
            texts = [classname]
            texts = clip.tokenize(texts)
            texts = jt.array(texts)
            class_embeddings = clip_model.encode_text(texts)
            
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

            del classname
            del class_embeddings
            jt.gc()
        
        clip_weights = jt.stack(clip_weights, dim=1)
        all_clip_weights.append(clip_weights)

    # 将所有提示词文件的权重求平均
    final_clip_weights = jt.mean(jt.stack(all_clip_weights, dim=2), dim=2)
    return final_clip_weights






def load_aux_weight(args, model, train_loader_cache, tfm_norm):
    jt.flags.use_cuda = 1
    if not args.load_aux_weight:
        aux_features = []
        aux_labels = []
        for augment_idx in range(args.augment_epoch):
            aux_features_current = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx, args.augment_epoch))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images =  jt.array(images)
                image_features = model(tfm_norm(images))
                aux_features_current.append(image_features)
                if augment_idx == 0:
                    target =  jt.array(target)
                    aux_labels.append(target)
                del images
                del target
                del image_features
                jt.gc()  # 强制释放显存
                jt.sync_all()
            aux_features.append(jt.concat(aux_features_current, dim=0).unsqueeze(0))
            jt.gc()  # 强制释放显存
            
        # aux_features =  jt.array(aux_features)
        aux_features = jt.concat(aux_features, dim=0).mean(dim=0)
        aux_features /= aux_features.norm(dim=-1, keepdim=True)
        # aux_labels =  jt.array(aux_labels)
        aux_labels = jt.concat(aux_labels)
        aux_features = aux_features.numpy()
        aux_labels = aux_labels.numpy()
        print("下面输出维度", aux_features.shape, aux_labels.shape,type(aux_features),type(aux_labels))
        jt.save(aux_features, args.cache_dir + f'/aux_feature_' + str(args.shots) + "shots.pkl")
        jt.save(aux_labels, args.cache_dir + f'/aux_labels_' + str(args.shots) + "shots.pkl")
    else:
        aux_features = jt.load(args.cache_dir + f'/aux_feature_' + str(args.shots) + "shots.pkl")
        aux_labels = jt.load(args.cache_dir + f'/aux_labels_' + str(args.shots) + "shots.pkl")
        print("下面输出维度", aux_features.shape, aux_labels.shape,type(aux_features),type(aux_labels))
    
    return aux_features, aux_labels

def build_cache_model(args, clip_model, train_loader_cache, tfm_norm):
    jt.flags.use_cuda = 1    
    if not args.load_cache:
        cache_keys = []
        cache_values = []
    
        # Data augmentation for the cache model
        for augment_idx in range(10):
            train_features = []
            print('Augment Epoch: {:} / {:}'.format(augment_idx, 10))
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
    
                images =  jt.array(images)
                image_features = clip_model.encode_image(tfm_norm(images))
                train_features.append(image_features)
                if augment_idx == 0:
                    target =  jt.array(target)
                    cache_values.append(target)
                del images
                del target
                del image_features
                jt.gc()  # 强制释放显存
            # train_features =  jt.array(train_features)
            cache_keys.append(jt.concat(train_features, dim=0).unsqueeze(0))
            jt.gc()  # 强制释放显存
        cache_keys = jt.concat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = jt.nn.one_hot(jt.concat(cache_values, dim=0)).half()
        print("下面输出维度", cache_keys.shape, cache_values.shape,type(cache_keys),type(cache_values))
        # cache_keys =  jt.array(cache_keys)
        # cache_values =  jt.array(cache_values)
        jt.save(cache_keys, args.cache_dir + '/keys_' + str(args.shots) + "shots.pkl")
        jt.save(cache_values, args.cache_dir + '/values_' + str(args.shots) + "shots.pkl")
    else:
        cache_keys = jt.load(args.cache_dir + '/keys_' + str(args.shots) + "shots.pkl")
        cache_values = jt.load(args.cache_dir + '/values_' + str(args.shots) + "shots.pkl")
        cache_keys =  jt.array(cache_keys)
        cache_values =  jt.array(cache_values)
        cache_keys = cache_keys.squeeze()
        cache_values = cache_values.squeeze()
        print("下面输出维度", cache_keys.shape, cache_values.shape,type(cache_keys),type(cache_values))
    
    return cache_keys, cache_values

def load_test_features(args, split, model, loader, tfm_norm, model_name):
    jt.flags.use_cuda = 1
    if not args.load_pre_feat:
        features, labels = [], []
        for i, (images, target) in enumerate(tqdm(loader)):

            if hasattr(model, 'encode_image') and callable(getattr(model, 'encode_image')):
                image_features = model.encode_image(tfm_norm(images))  # for clip model
            else:
                image_features = model(tfm_norm(images))
            features.append(image_features)

            labels.append(target)

            del images
            del target
            del image_features
            jt.gc()  # 强制释放显存
            jt.sync_all()
        features, labels = jt.concat(features), jt.concat(labels)

        print("下面输出维度", features.shape, labels.shape,type(features),type(labels))

        jt.save(features, args.cache_dir + f"/{model_name}_" + split + "_f.pkl")
        jt.save(labels, args.cache_dir + f"/{model_name}_" + split + "_l.pkl")

    else:
        features = jt.load(args.cache_dir + f"/{model_name}_" + split + "_f.pkl")
        labels = jt.load(args.cache_dir + f"/{model_name}_" + split + "_l.pkl")
        features =  jt.array(features)
        labels =  jt.array(labels)
        print("下面输出维度", features.shape, labels.shape,type(features),type(labels))
    return features, labels



def config_logging(args):
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M')
    now = datetime.datetime.now().strftime("%m-%d-%H_%M")
    modified_string = args.dataset.replace('/', '-')
    split_words = modified_string.split('-')
    end_name = '-'.join(split_words[-2:])
    fh = logging.FileHandler(f'result/{end_name}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger



