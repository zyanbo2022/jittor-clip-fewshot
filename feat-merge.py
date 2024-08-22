import jittor as jt
import os
import pdb
import numpy as np
def mean_merge(feats):
    mean_feat = jt.mean(feats, dim=0)
    mean_feat /= mean_feat.norm(dim=-1, keepdim=True)
    return mean_feat

def load_feature(dir, scale, split, name):

    feat_dir = dir + "/" + name + "_"+ split + "_f_" + scale + ".pkl"
    features = jt.load(feat_dir)
    features =  jt.array(features)
    return features


def save_feature(feat, save_dir, split, name):

    save_path = save_dir + "/" +name+"_"+ split + "_f.pkl"
    jt.save(feat, save_path)



textual_dir = os.path.join('./caches', 'trainset0')
feat_dir = os.path.join('./caches', 'selected')
save_dir = os.path.join('./caches', 'trainset')
global_dir = os.path.join('./caches', 'trainset0')
# textual features
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



features_all = jt.zeros(load_feature(feat_dir, '5', 'val','clip').shape).half() #.cuda()

for scale in range(1, 11):
    # test features
    print("Getting val features.")
    print(scale)
    if scale != 10:
        features = load_feature(feat_dir, str(scale), 'val','clip')
    else:
        features = load_feature(global_dir, str(scale), 'val','clip')
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.squeeze(0)
    ratio = scale / 55.0
    
    features_all += features * ratio

features_all /= features_all.norm(dim=-1, keepdim=True)
save_feature(features_all, save_dir, 'val','clip')



# del features_all
features_all = jt.zeros(load_feature(feat_dir, '5', 'test','clip').shape).half() #.cuda()

for scale in range(1, 11):
    # test features
    print("Getting test features.")
    print(scale)
    if scale != 10:  
        features = load_feature(feat_dir, str(scale), 'test','clip')
    else:
        features = load_feature(global_dir, str(scale), 'test','clip')
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.squeeze(0)
    ratio = scale / 55.0
    
    features_all += features * ratio

features_all /= features_all.norm(dim=-1, keepdim=True)
save_feature(features_all, save_dir, 'test','clip')




