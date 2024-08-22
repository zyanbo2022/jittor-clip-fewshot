import jittor as jt
import jittor.nn as nn
import os
from jittor.models import resnet50
# https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md
def load_moco(pretrain_path):
    print("=> creating model")
    model = resnet50()
    linear_keyword = 'fc'
    if os.path.isfile(pretrain_path):
        print("=> loading checkpoint '{}'".format(pretrain_path))
        state_dict = jt.load(pretrain_path)  # 直接加载state_dict

        for k in list(state_dict.keys()):
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        # 获取模型的当前参数字典
        model_state_dict = model.state_dict()

        # 更新模型的参数字典
        model_state_dict.update(state_dict)

        # 手动加载参数
        model.load_parameters(model_state_dict)

        # 找出缺失和多余的键
        missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())

        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        assert missing_keys == {f"{linear_keyword}.weight", f"{linear_keyword}.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrain_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_path))
        raise FileNotFoundError
    model.fc = nn.Identity()
    return model, 2048

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    jt.flags.use_cuda = 1
    model, dim = load_moco("../r-50-1000ep.pkl")
    # print(model)
    num_params = count_parameters(model)
    num_params_in_millions = num_params / 1_000_000
    print(f"Number of parameters in millions: {num_params_in_millions:.2f}M")
    # 23.56M
