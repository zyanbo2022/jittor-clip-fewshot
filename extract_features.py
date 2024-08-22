import os
import random
import json
from tqdm import tqdm
import jittor as jt
from jittor import nn, transform as jt_transform

from run_utils import *
from datasets.TrainSet import TrainSet_double

from clip.amu import AMU_Model ,tfm_clip,tfm_aux
from clip.moco import load_moco

from clip import clip
from parse_args import parse_args
from utils import *
from datasets.utils import DatasetWrapper
import numpy
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora

if __name__ == '__main__':
    jt.flags.use_cuda = 1

    # Load config file
    parser = parse_args()
    args = parser.parse_args()
    argslora = get_arguments()
    cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir
    args.load_pre_feat =False
    args.load_aux_weight=False
    args.load_cache=False

    args_dict = vars(args)
    message = '\n'.join([f'{k:<20}: {v}' for k, v in args_dict.items()])
    print(message)

    clip_model, preprocess = clip.load('ViT-B-32.pkl')#ViT-B-32.pkl
    clip_model.eval()
    
    list_lora_layers = apply_lora(argslora, clip_model)
    load_lora(argslora, list_lora_layers)
    clip_model.eval()

    aux_model, args.feat_dim = load_moco("r-50-1000ep.pkl")

    aux_model.eval()


    random.seed(args.rand_seed)
    numpy.random.seed(args.rand_seed)
    jt.set_global_seed(args.rand_seed)
    jt.seed(args.rand_seed)
    jt.set_seed(args.rand_seed)

    dataset = TrainSet_double()
    train_loader = DatasetWrapper(data_source=dataset.train_x, batch_size=32, tfm=tfm_train_base, is_train=True, shuffle=False)
    val_loader = DatasetWrapper(data_source=dataset.val, batch_size=80, is_train=False, tfm=tfm_test_base, shuffle=False)
    test_loader = DatasetWrapper(data_source=dataset.test, batch_size=80, is_train=False, tfm=tfm_test_base, shuffle=False)



    print("Constructing  aux_cache  mode")
    aux_features, aux_labels = load_aux_weight(args, aux_model, train_loader, tfm_norm=tfm_aux)


    print("Constructing tip_cache  mode")
    cache_keys, cache_values = build_cache_model(args, clip_model, train_loader, tfm_norm=tfm_clip)


    # print("Loading clip features and labels from val set.")
    # val_clip_features, val_labels = load_test_features(args, "val", clip_model, val_loader, tfm_norm=tfm_clip, model_name='clip')
    print("Loading aux features and labels from val set.")
    val_aux_features, val_labels = load_test_features(args, "val", aux_model, val_loader, tfm_norm=tfm_aux, model_name='aux')

    # print("Loading clip features and labels from test set.")
    # # test_clip_features, test_labels = load_test_features(args, "test", clip_model, test_loader, tfm_norm=tfm_clip, model_name='clip')
    print("Loading aux features and labels from test set.")
    test_aux_features, test_labels = load_test_features(args, "test", aux_model, test_loader, tfm_norm=tfm_aux, model_name='aux')
    print("------全部提取完毕------")