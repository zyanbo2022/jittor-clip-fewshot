import jittor as jt
import yaml
import os
import pdb
from clip.amu import Linear_Adapter,logit_normalize,tfm_aux
from parse_args import parse_args
from datasets.TrainSet import TrainSet_double
from datasets.utils import DatasetWrapper
from utils import tfm_train_base
from clip.moco import load_moco
from utils import *

def load_text_feature374(textual_dir):
    save_path = textual_dir + "/text_weights374.pkl"
    clip_weights = jt.load(save_path)
    clip_weights =  jt.array(clip_weights)
    print('clip_weights地址',save_path)
    return clip_weights

def load_text_feature403(textual_dir):
    save_path = textual_dir + "/text_weights403.pkl"
    clip_weights = jt.load(save_path)
    clip_weights =  jt.array(clip_weights)
    print('clip_weights地址',save_path)
    return clip_weights

def select_image_views(view_feat, clip_weights):
    norm_view_feat = view_feat / view_feat.norm(dim=-1, keepdim=True) #norm_view_feat Size([10, 15226, 512])
    local_logits = norm_view_feat @ clip_weights   #local_logits.shape Size([10, 15226, 374])
    logits_values, _ = jt.topk(local_logits, k=2, dim=-1) #[10,15226,2,]
    criterion = logits_values[:,:,0] - logits_values[:,:,1] #Size([10, 15226])
    local_idx = jt.argsort(criterion, dim=0, descending=True)[0][:1] #[1,15226,]

    # selected = take_along_dim(view_feat, local_idx[:,:,None], dim=0).squeeze(0)
    expanded_idx = jt.unsqueeze(local_idx, -1) # 
    
    # 将 expanded_idx 扩展到与 view_feat 的形状匹配
    expanded_idx = jt.concat([expanded_idx] * view_feat.shape[-1], dim=-1)#[1,15226,512,]
    
    # 使用 gather 操作提取元素
    selected = jt.gather(view_feat, dim=0, index=expanded_idx) #[10,15226,512,]
    
    # 如果第 0 维的大小为 1，则压缩该维度
    if selected.shape[0] == 1:#[10,15226,512,]
        selected = selected.squeeze(0)
        

    return selected # Size([15226, 512])




def load_feature(dir, scale, split, name):
    feat_dir = dir + "/" +name+"_"+ split + "_f_" + scale + ".pkl"
    features = jt.load(feat_dir)
    features =  jt.array(features)
    # cache_values =  jt.array(cache_values)
    
    return features

def save_feature(selected_features, save_dir, scale, split,name):
    save_path = save_dir + "/" +name+"_"+ split +"_f_" + scale +".pkl"
    print('savepath',save_path)
    jt.save(selected_features, save_path)
    print(save_path,'处理成功')
    return

parser = parse_args()
args = parser.parse_args()
cache_dir = os.path.join('./caches', args.dataset)
os.makedirs(cache_dir, exist_ok=True)
args.cache_dir = cache_dir
dataset = TrainSet_double()
train_loader = DatasetWrapper(data_source=dataset.train_x, batch_size=32, tfm=tfm_train_base, is_train=True, shuffle=False)

textual_dir = os.path.join('./caches', 'trainset0')
feat_dir = os.path.join('./caches', 'trainset0')
save_dir = os.path.join('./caches', 'selected')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# textual features
print("Getting textual features as CLIP's classifier.")
clip_weight403 = load_text_feature403(textual_dir)
clip_weight374 = load_text_feature374(textual_dir)

aux_model, feat_dim = load_moco("r-50-1000ep.pkl")
aux_model.eval()

print(" load  aux_features...")
aux_features, aux_labels = load_aux_weight(args, aux_model, train_loader, tfm_norm=tfm_aux)
sample_features=[aux_features, aux_labels]

aux_adapter374 = Linear_Adapter(feat_dim, 374, sample_features=sample_features)
aux_adapter403 = Linear_Adapter(feat_dim, 403, sample_features=sample_features)

for scale in range(1, 10):
    print(f"\nProcessing : {scale}")
    
    print("Getting clip val features.")
    features = load_feature(feat_dir, str(scale), 'val','clip')
    selected_feature = select_image_views(features, clip_weight374)
    save_feature(selected_feature, save_dir, str(scale), 'val','clip')

    print("Getting clip test  features.")
    features = load_feature(feat_dir, str(scale), 'test','clip')
    selected_feature = select_image_views(features, clip_weight403)
    save_feature(selected_feature, save_dir, str(scale), 'test','clip')

  