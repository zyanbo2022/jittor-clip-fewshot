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

def freeze_bn(m):
    classname = m.__class__.__name__
    if 'BatchNorm' in classname:
        m.eval()

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
    
def train_one_epoch(model, data_loader, optimizer, scheduler, logger):
    # Train
    model.train()
    model.apply(freeze_bn)  # freeze BN-layer
    correct_samples, all_samples = 0, 0
    loss_list = []
    loss_aux_list = []
    loss_merge_list = []

    for i, (images, target) in enumerate(tqdm(data_loader)):

        return_dict = model(images, labels=target)

        acc = cls_acc(return_dict['logits'], target)
        correct_samples += acc / 100 * len(return_dict['logits'])
        all_samples += len(return_dict['logits'])
        
        loss_list.append(return_dict['loss'].item())
        loss_aux_list.append(return_dict['loss_aux'].item())
        loss_merge_list.append(return_dict['loss_merge'].item())

        optimizer.zero_grad()
        optimizer.backward(return_dict['loss'])
        optimizer.step()
        scheduler.step()
        jt.sync_all()


    logger.info('Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))
    logger.info("Loss_aux: {:.4f}, Loss_merge: {:.4f}".format(
        sum(loss_aux_list)/len(loss_aux_list), sum(loss_merge_list)/len(loss_merge_list)))

def train_and_eval(args, dataset,logger, model, clip_test_features, aux_test_features, test_labels, clip_val_features, aux_val_features, val_labels, train_loader_F):
    jt.flags.use_cuda = 1
    # model.requires_grad_(False)
    model.aux_adapter.requires_grad_(True)
    # model.tipadapter.requires_grad_(True)
    for name, param in model.named_parameters():
        if param.requires_grad :
            print(name, param.requires_grad)

    optimizer = jt.optim.AdamW(
        # model.parameters(),
        [
    {"params": model.aux_adapter.parameters()}],
        weight_decay=0.01,

        lr=args.lr,
        eps=1e-4
    )


    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch * len(train_loader_F))

    best_acc, best_epoch = 0.0, 0

    for train_idx in range(1, args.train_epoch + 1):
        logger.info('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
        train_one_epoch(model, train_loader_F, optimizer, scheduler, logger)
        # Eval
        model.eval()
        with jt.no_grad():
            return_dict = model(
                clip_features=clip_val_features,
                aux_features=aux_val_features,
                labels=val_labels
            )
            all_logits = return_dict['logits']
            aux_logits = return_dict['aux_logits']


            acc = cls_acc(all_logits, val_labels)
            acc_aux = cls_acc(aux_logits, val_labels)


        logger.info("----- aux  val Acc: {:.4f} ----".format(acc_aux))
        logger.info("----- all  val Acc: {:.4f} -----".format(acc))

        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            logger.info("-----开始保存权重----- ")
            jt.save(model.aux_adapter.fc.weight, args.cache_dir + "/best_adapter_" + str(args.shots) + "shots.pkl")

            
    logger.info(f"----- Best Test Acc: {best_acc:.4f}, at epoch: {best_epoch}.-----\n")

    

if __name__ == '__main__':
    jt.flags.use_cuda = 1

    # Load config file
    parser = parse_args()
    args = parser.parse_args()
    argslora = get_arguments()
    cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir
    args.train_epoch = 20
    logger = config_logging(args)
    logger.info("\nRunning configs.")
    args_dict = vars(args)
    message = '\n'.join([f'{k:<20}: {v}' for k, v in args_dict.items()])
    logger.info(message)


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

    
    logger.info("Loading  dataset....")
    dataset = TrainSet_double()
    train_loader = DatasetWrapper(data_source=dataset.train_x, batch_size=32, tfm=tfm_train_base, is_train=True, shuffle=False)
    train_loader_shuffle = DatasetWrapper(data_source=dataset.train_x, batch_size=32, tfm=tfm_train_base, is_train=True, shuffle=True)

    val_loader = DatasetWrapper(data_source=dataset.val, batch_size=80, is_train=False, tfm=tfm_test_base, shuffle=False)
    test_loader = DatasetWrapper(data_source=dataset.test, batch_size=80, is_train=False, tfm=tfm_test_base, shuffle=False)


    logger.info("Getting textual features as CLIP's classifier...")
    classnames = open('prompt/b.txt').read().splitlines()
    clip_weights = clip_classifier(classnames,clip_model)

    logger.info(" load  aux_features...")
    aux_features, aux_labels = load_aux_weight(args, aux_model, train_loader, tfm_norm=tfm_aux)


    logger.info("Loading clip features and labels from val set.")
    val_clip_features, val_labels = load_test_features(args, "val", clip_model, val_loader, tfm_norm=tfm_clip, model_name='clip')
    
    logger.info("Loading aux features and labels from val set.")
    val_aux_features, val_labels = load_test_features(args, "val", aux_model, val_loader, tfm_norm=tfm_aux, model_name='aux')

    val_clip_features = jt.array(val_clip_features)
    val_labels = jt.array(val_labels)
    val_aux_features =  jt.array(val_aux_features)

    logger.info("Loading clip features and labels from test set.")
    test_clip_features, test_labels = load_test_features(args, "test", clip_model, test_loader, tfm_norm=tfm_clip, model_name='clip')
    
    logger.info("Loading aux features and labels from test set.")
    test_aux_features, test_labels = load_test_features(args, "test", aux_model, test_loader, tfm_norm=tfm_aux, model_name='aux')

    test_clip_features =  jt.array(test_clip_features)
    test_aux_features = jt.array(test_aux_features)



    val_clip_features = val_clip_features.astype(jt.float32)
    clip_weights = clip_weights.astype(jt.float32)

    args.num_classes = 403
    print(args.num_classes)
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

    train_and_eval(args, dataset, logger, model, test_clip_features, test_aux_features, test_labels, val_clip_features, val_aux_features, val_labels, train_loader_shuffle)
    
