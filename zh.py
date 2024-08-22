import torch
import jittor as jt

# 加载预训练模型的checkpoint
checkpoint = torch.load('r-50-1000ep.pth.tar')

# 检查加载对象的类型并获取state_dict
if 'state_dict' in checkpoint:
    clip = checkpoint['state_dict']
else:
    raise ValueError("Checkpoint does not contain 'state_dict'")

# 将每个参数转换为float类型并移动到CPU
for k in clip.keys():
    clip[k] = clip[k].float().cpu().numpy()  # 转换为numpy数组以兼容Jittor

# 保存转换后的state_dict
jt.save(clip, 'r-50-1000ep.pkl')


# import torch
# import jittor as jt
# clip = torch.load('ViT-B-32.pt').state_dict()

# for k in clip.keys():
#     clip[k] = clip[k].float().cpu()
# jt.save(clip, 'ViT-B-32.pkl')
