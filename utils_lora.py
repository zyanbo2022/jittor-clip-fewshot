from tqdm import tqdm
import jittor as jt
import clip
from PIL import Image
from jittor import transform as jt_transform
from jittor.transform import Compose, ImageNormalize


def cls_acc(output, target):


    l_argmax = output.argmax(dim=-1)
    argmax_result = l_argmax[0]

    correct_predictions = (argmax_result == target).sum()
    acc = 100 * correct_predictions/ target.shape[0]
    return acc




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



def clip_classifier(classnames, template, clip_model):
    clip_weights = []
    jt.flags.use_cuda = 1
    with jt.no_grad():
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

    
tfm_clip = Compose([ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
def pre_load_features(clip_model, loader):
    features, labels = [], []
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            image_features = clip_model.encode_image(tfm_clip(images))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
            del image_features
            jt.gc()
            jt.sync_all()
        features, labels = jt.cat(features), jt.cat(labels)
    
    return features, labels

