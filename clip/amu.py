import jittor as jt
import jittor.nn as nn
from jittor.transform import Compose, ImageNormalize
from clip.focalloss import gce_loss

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
    def __init__(self, clip_model, aux_model, sample_features, clip_weights, feat_dim, class_num, lambda_merge, alpha):
        super().__init__()

        

        self.clip_model = clip_model
        self.aux_model = aux_model
        self.clip_weights = clip_weights
        self.aux_adapter = Linear_Adapter(feat_dim, class_num, sample_features=sample_features)
        # self.aux_adapter.fc.weight.requires_grad = True
        self.lambda_merge = lambda_merge
        self.alpha = alpha

        
    def execute(self, images=None, clip_features=None, aux_features=None, labels=None):
        if images is not None:
            clip_features, aux_features = self.forward_feature(images)

            
        clip_features = clip_features / clip_features.norm(dim=-1, keepdims=True)
        aux_features = aux_features / aux_features.norm(dim=-1, keepdims=True)
        clip_logits, aux_logits = self.forward_adapter(clip_features, aux_features)

        logits = clip_logits + aux_logits * self.alpha  

        if labels is not None:

            loss_merge = gce_loss(logits, labels)
            loss_aux = gce_loss(aux_logits, labels)
            loss = self.lambda_merge * loss_merge + (1 - self.lambda_merge) * loss_aux

        else:
            loss_merge = None
            loss_aux = None
            loss = None

        aux_logits = clip_logits +  aux_logits * self.alpha 

        return_dict = {
            "logits": logits,
            "clip_logits": clip_logits,
            "aux_logits": aux_logits,
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
    
    def forward_adapter(self, clip_features, aux_features):
        # logits
        clip_logits = 100. * clip_features @ self.clip_weights
        aux_logits = self.aux_adapter(aux_features)
        
        aux_logits = logit_normalize(aux_logits)

        return clip_logits, aux_logits
