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



def write_top5_results_to_txt(logits, output_file, image_names):
    
    top5_indices = jt.argsort(logits, dim=1, descending=True)[0].numpy()[:,:5]
    with open(output_file, 'w') as f:
        for i in range(len(image_names)):
            image_name = image_names[i]
            top5_classes = top5_indices[i].tolist()
            f.write(f"{image_name[9:]} {' '.join(map(str, top5_classes))}\n")


def get_image_names_from_txt(txt_file):
    image_names = []
    with open(txt_file, 'r') as f:
        for line in f:
            columns = line.strip().split()
            if columns:  # 确保行不为空
                image_names.append(columns[0])
    return image_names
    


if __name__ == '__main__':
    
    jt.flags.use_cuda = 1

    # Load config file
    parser = parse_args()
    args = parser.parse_args()
    argslora = get_arguments()
    cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir


    clip_model, preprocess = clip.load('ViT-B-32.pkl')
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
    train_loader_shuffle = DatasetWrapper(data_source=dataset.train_x, batch_size=32, tfm=tfm_train_base, is_train=True, shuffle=True)

    val_loader = DatasetWrapper(data_source=dataset.val, batch_size=80, is_train=False, tfm=tfm_test_base, shuffle=False)
    test_loader = DatasetWrapper(data_source=dataset.test, batch_size=80, is_train=False, tfm=tfm_test_base, shuffle=False)


    print("Getting textual features as CLIP's classifier...")
    classnames = open('prompt/b.txt').read().splitlines()
    clip_weights = clip_classifier(classnames,clip_model)

    print(" load  aux_features...")
    aux_features, aux_labels = load_aux_weight(args, aux_model, train_loader, tfm_norm=tfm_aux)
    

    print("Loading  features and labels from val set.")
    val_clip_features, val_labels = load_test_features(args, "val", clip_model, val_loader, tfm_norm=tfm_clip, model_name='clip')
    val_aux_features, val_labels = load_test_features(args, "val", aux_model, val_loader, tfm_norm=tfm_aux, model_name='aux')
    
    val_clip_features = jt.array(val_clip_features)
    val_labels = jt.array(val_labels)
    val_aux_features =  jt.array(val_aux_features)

    print("Loading  features and labels from test set.")
    test_clip_features, test_labels = load_test_features(args, "test", clip_model, test_loader, tfm_norm=tfm_clip, model_name='clip')
    test_aux_features, test_labels = load_test_features(args, "test", aux_model, test_loader, tfm_norm=tfm_aux, model_name='aux')

    test_clip_features =  jt.array(test_clip_features)
    test_aux_features = jt.array(test_aux_features)


    val_clip_features = val_clip_features.astype(jt.float32)
    clip_weights = clip_weights.astype(jt.float32)
    test_clip_features = test_clip_features.astype(jt.float32)
    test_aux_features = jt.array(test_aux_features)


    args.num_classes = 403
    model = AMU_Model(
        clip_model=clip_model,
        aux_model=aux_model,
        sample_features=[aux_features, aux_labels],
        clip_weights=clip_weights,
        feat_dim=args.feat_dim,
        class_num=args.num_classes,
        lambda_merge=args.lambda_merge,
        alpha=args.alpha
        )
    
    bestpkl = jt.load(args.cache_dir + "/best_adapter_" + str(args.shots) + "shots.pkl")
    model.aux_adapter.fc.weight =  jt.array(bestpkl)
    model.eval()
    # with jt.no_grad():
        
    #     return_dict = model(
    #         clip_features=val_clip_features,
    #         aux_features=val_aux_features,
    #         labels=val_labels
    #     )
    #     all_logits = return_dict['logits']
    #     aux_logits = return_dict['aux_logits']


    #     acc = cls_acc(all_logits, val_labels)
    #     acc_aux = cls_acc(aux_logits, val_labels)


    # print("----- aux  val Acc: {:.4f} ----".format(acc_aux))
    # print("----- all  val Acc: {:.4f} -----".format(acc))
    
    with jt.no_grad():
        return_dict = model(
            clip_features=test_clip_features,
            aux_features=test_aux_features,
        )
        
        all_logits = return_dict['logits']
        image_names = get_image_names_from_txt('caches/testb.txt')
        write_top5_results_to_txt(all_logits,'result/result.txt', image_names)
        print("-----测试完成，结果保存在result/result.txt-----\n ")
            

    
