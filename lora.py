import jittor as jt
from jittor import transform as jt_transform
import jittor.nn as nn
from utils_lora import *
import json
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
# from loralib import layers as lora_layers
from jittor.transform import Compose, ImageNormalize



tfm_clip = Compose([ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
def write_top5_results_to_txt(logits, output_file, image_names, batch_id):
    # 获取前五个最大值的索引
    top5_indices = jt.argsort(logits, dim=1, descending=True)[0].numpy()[:,:5]

    with open(output_file, 'a') as f:
        if batch_id==30:
            for i in range(73):
                image_name = image_names[i+(batch_id*100)]
                top5_classes = top5_indices[i].tolist()
                f.write(f"{image_name[9:]} {' '.join(map(str, top5_classes))}\n")
        else: 
            for i in range(100):
                image_name = image_names[i+(batch_id*100)]
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


def test_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with jt.no_grad():
        template = dataset.template[0]
        classnames = open('prompt/a.txt').read().splitlines()
        texts = [classname for classname in classnames]
        # texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
        # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        texts = clip.tokenize(texts)#.cuda()
        class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with jt.no_grad():
        for i, (images, target) in enumerate(loader):
            # images, target = images.cuda(), target.cuda()
            # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(tfm_clip(images))
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()

            image_names = get_image_names_from_txt('caches/test.txt')
            write_top5_results_to_txt(cosine_similarity,'caches/res.txt', image_names, i)
            del image_features
            del cosine_similarity
            jt.sync_all()
            jt.gc()
        print('写入完成')
            


def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with jt.no_grad():
        template = dataset.template[0]
        classnames = open('prompt/a.txt').read().splitlines()
        texts = [classname for classname in classnames]
        # texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
        # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        texts = clip.tokenize(texts)#.cuda()
        class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    jt.flags.use_cuda = 1
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            # images, target = images.cuda(), target.cuda()
            # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(tfm_clip(images))
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity) 
            del image_features
            del cosine_similarity
            jt.sync_all()
            jt.gc()
    acc /= tot_samples
    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = True
    jt.flags.use_cuda = 1
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    classnames = open('prompt/a.txt').read().splitlines()
    textual_features = clip_classifier(classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    val_features = jt.array(val_features)
    val_labels = jt.array(val_labels)
    val_features = val_features.astype(jt.float32)
    textual_features = textual_features.astype(jt.float32)

    
    # Zero-shot CLIP
    
    clip_logits = logit_scale * val_features @ textual_features
    zs_acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.4f}. ****\n".format(zs_acc))

    val_features = val_features.cpu()
    val_labels = val_labels.cpu()
    
    jt.sync_all()
    jt.gc()
    
    
    list_lora_layers = apply_lora(args, clip_model)
    # print(list_lora_layers)
    # clip_model = clip_model.cuda()
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
        print("**** val accuracy: {:.4f}. ****\n".format(acc_val))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = jt.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # training LoRA
    # scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            template = dataset.template[0]
            classnames = open('prompt/a.txt').read().splitlines()
            texts = [classname for classname in classnames]
            
            # print(texts)
            # texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
            # images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = clip.tokenize(texts) #.cuda()
                class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            else:
                with jt.no_grad():
                    # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = nn.cross_entropy_loss(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            scheduler.step()
            scheduler.update_lr()
            count_iters += 1
            del image_features
            del cosine_similarity
            jt.sync_all()
            jt.gc()
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            # breakpoint()
            # current_lr = scheduler.get_lr()[0]
            print('lr: {:.8f} Acc: {:.4f}, Loss: {:.4f}'.format(optimizer.lr,acc_train, loss_epoch))
            print(count_iters,total_iters)

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("****  val accuracy: {:.4f}. ****\n".format(acc_val))
            if acc_val > best_acc_val:
                print('开始测试')
                # test_lora(args, clip_model, test_loader, dataset)
                best_acc_val = acc_val
                save_lora(args, list_lora_layers)
                print('loar已保存')
            