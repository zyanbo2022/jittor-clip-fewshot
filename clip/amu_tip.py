import jittor as jt
import jittor.nn as nn
from jittor.transform import Compose, ImageNormalize
from clip.focalloss import gce_loss
import clip
# def compute_scores(val_clip_features, clip_model, file_paths):
#     scores = []
#     for file_path in file_paths:
#         classnames = open(file_path).read().splitlines()
#         clip_weights = clip_classifier(classnames, clip_model)
#         clip_weights = clip_weights.astype(jt.float32)
        
#         # tmp = val_clip_features / val_clip_features.norm(dim=-1, keepdim=True)
#         score = 100. * val_clip_features @ clip_weights
#         scores.append(score)
    
#     return scores

# def combine_scores(scores, alphas):
#     if len(scores) != len(alphas):
#         raise ValueError("Number of scores must match the number of alphas")
    
#     combined_score = alphas[0] * scores[0]
#     for i in range(1, len(scores)):
#         combined_score += alphas[i] * scores[i]
    
#     return combined_score
# def clip_classifier(classnames,  clip_model):
#     clip_weights = []
#     jt.flags.use_cuda = 1
#     for classname in classnames:
#         texts = [classname]
#         texts = clip.tokenize(texts)
#         texts =  jt.array(texts)
#         class_embeddings = clip_model.encode_text(texts)

        
#         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#         class_embedding = class_embeddings.mean(dim=0)
#         class_embedding /= class_embedding.norm()
#         clip_weights.append(class_embedding)

#         del classname
#         del class_embeddings
#         jt.gc()
#     clip_weights = jt.stack(clip_weights, dim=1)
#     return clip_weights
def std_along_dim(x, dim, keepdims=False):
    
    mean = jt.mean(x, dim=dim, keepdims=True)
    diff = x - mean
    sqr = diff * diff
    variance = jt.mean(sqr, dim=dim)
    std = jt.sqrt(variance)
    if not keepdims:
        std = jt.squeeze(std, dim=dim)
    return std
    
def logit_normalize(logit):
    
    logit = jt.array(logit)
    logits_std = std_along_dim(logit, dim=1, keepdims=True)
    # logits_std = jt.std(logit, dim=1, keepdims=True)
    logits_mean = jt.mean(logit, dim=1, keepdims=True)
    # print(logit.shape,logits_mean.shape,logits_std.shape,type(logit),type(logits_mean),type(logits_std))
    logits_std_expanded = logits_std.unsqueeze(1)
    logit = (logit - logits_mean) / logits_std_expanded
    return logit

def uncertainty(logits, type, power):
    softmax_fun = nn.Softmax(dim=-1) # softmax to get probability distribution
    logits = softmax_fun(logits)
    if type == 'entropy':
        entropy = -jt.sum(logits * jt.log2(logits), dim=-1, keepdims=True) / jt.log2(jt.array(logits.shape[-1]).float32())
        entropy =  (entropy * power).exp() 
        return entropy
    elif type == 'energy':
        max_values = logits.max(dim=-1, keepdims=True)
        logits = logits - max_values
        tau = 2
        energy = tau * (jt.log(jt.sum(jt.exp(logits / tau), dim=-1, keepdims=True)) + max_values)
        return 1.0 / (energy ** power)
    elif type == 'max':
        max_values = logits.max(dim=-1, keepdims=True)
        return 1.0 / (max_values) ** power
    elif type == 'max-min':
        diff = logits.max(dim=-1, keepdims=True) - logits.min(dim=-1, keepdims=True)
        return 1.0 / diff ** power 
    elif type == 'var':
        variance = jt.std(logits, dim=-1, keepdims=True)
        return variance
    elif type == 'top5':
        top2 = logits.topk(5, dim=-1)[0]
        confidence = (top2[:, 0] - top2[:, -1]).unsqueeze(-1)
        return 1.0 / (confidence) ** power
    elif type == 'moment':
        mu = jt.mean(logits, dim=-1, keepdims=True)
        sigma = std_along_dim(logits, dim=1, keepdims=True)
        sigma = sigma.unsqueeze(1)
        # sigma = jt.std(logits, dim=-1, keepdims=True)
        normalized_logits = (logits - mu) / sigma
        moment_4 = jt.mean(normalized_logits ** 4, dim=-1, keepdims=True)
        return 1 / ((moment_4 / 250) ** power)
    elif type == 'none':
        return jt.array(1.0)
    else:
        raise RuntimeError('Invalid uncertainty type.')

class Linear_Adapter(nn.Module):
    def __init__(self, feat_dim, class_num, sample_features=None):
        super().__init__()
        self.fc = nn.Linear(feat_dim, class_num, bias=False)
        # init
        if sample_features is not None:
            print('init adapter weight by training samples...')
            aux_features, aux_labels = sample_features[0], sample_features[1]
            aux_features = aux_features

            init_weight = jt.zeros(feat_dim, class_num)
            #, device=aux_features.device) 
            # print("下面输出维度", aux_features.shape, aux_labels.shape, type(aux_features),type(aux_labels))
            for i in range(len(aux_labels)):
                init_weight[:, aux_labels[i]] += aux_features[i]
    
            feat_per_class = len(aux_labels) / class_num
            init_weight = init_weight / feat_per_class
            self.fc.weight = nn.Parameter(init_weight.t())
        else:
            print('init adapter weight by random...')
        
    def execute(self, feat):
        return self.fc(feat)

tfm_clip = Compose([ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
tfm_aux = Compose([ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class AMU_Model(nn.Module):
    def __init__(self, clip_model, cache_keys, cache_values, aux_model, sample_features,clip_weights, feat_dim, class_num, lambda_merge, alpha, alpha_c, uncent_type, uncent_power):
        super().__init__()
        self.tipadapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False)
        self.tipadapter.weight = nn.Parameter(cache_keys.t())
        self.tipadapter.weight = jt.array(self.tipadapter.weight, dtype=jt.float32)

        self.clip_model = clip_model
        self.aux_model = aux_model
        self.clip_weights = clip_weights
        # self.clip_weights1 = clip_weights1
        self.aux_adapter = Linear_Adapter(feat_dim, class_num, sample_features=sample_features)
        self.aux_adapter.fc.weight.requires_grad = True
        self.cache_values = cache_values
        self.lambda_merge = lambda_merge
        self.uncent_type = uncent_type
        self.uncent_power = uncent_power
        self.alpha = alpha
        self.alpha2 = alpha_c
        
    def execute(self, images=None, clip_features=None, aux_features=None, labels=None):
        if images is not None:
            clip_features, aux_features = self.forward_feature(images)

            
        clip_features = clip_features / clip_features.norm(dim=-1, keepdims=True)
        aux_features = aux_features / aux_features.norm(dim=-1, keepdims=True)
        clip_logits, aux_logits, cache_logits = self.forward_adapter(clip_features, aux_features, self.cache_values)

        # fusion
        factor = uncertainty(
            clip_logits.float32(),
            power=self.uncent_power,
            type=self.uncent_type
        )

        logits = clip_logits +  factor * aux_logits * self.alpha  + cache_logits * self.alpha2
        # logits = clip_logits + factor * aux_logits * self.alpha
        if labels is not None:
# nn.cross_entropy_loss
            loss_merge = gce_loss(logits, labels)
            loss_aux = gce_loss(aux_logits, labels)
            loss_tip = gce_loss(cache_logits, labels)
            # lambda_merge: 0.3
            loss = 0.3 * loss_merge + 3 * 0.3 * loss_aux + 1 * 0.3 * loss_tip

        else:
            loss_merge = None
            loss_aux = None
            loss = None
        cache_logits = clip_logits+ 2.0*cache_logits 
        aux_logits = clip_logits+ 0.5*aux_logits 
        # return logits, loss
        return_dict = {
            "logits": logits,
            "clip_logits": clip_logits,
            "aux_logits": aux_logits,
            "tip_logits": cache_logits,
            "loss": loss,
            "loss_merge": loss_merge,
            "loss_aux": loss_aux,            
        }
        return return_dict

    def forward_feature(self, images):
        # CLIP branch
        clip_features = self.clip_model.encode_image(tfm_clip(images))
        # AUX branch
        aux_feature = self.aux_model(tfm_aux(images))
        return clip_features, aux_feature
    
    def forward_adapter(self, clip_features, aux_features, cache_values):
        # logits
        clip_logits = 100. * clip_features @ self.clip_weights
        # clip_logits1 = 100. * clip_features @ self.clip_weights1
        # clip_logits = clip_logits1+0.22*clip_logits1
        # scores = compute_scores(clip_features, self.clip_model, self.file_paths)
        # alphas = [1, 0.22]
        # clip_logits = combine_scores(scores, alphas)
        aux_logits = self.aux_adapter(aux_features)
        aux_logits = logit_normalize(aux_logits)
        affinity = self.tipadapter(clip_features)

        a = ((-1) * (1.0 - 1.0 * affinity)).exp()
        cache_logits = a @ cache_values

        return clip_logits, aux_logits, cache_logits
