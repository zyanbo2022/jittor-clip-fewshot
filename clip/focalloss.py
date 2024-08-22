
import jittor as jt
from jittor import nn, Module

class GeneralizedCrossEntropyLoss(Module):
    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.q = q

    def execute(self, logits, targets):
        # 获取每个样本对应的预测概率
        probs = logits.softmax(dim=1)
        
        # 选择每个样本的真实标签对应的预测概率
        probs = probs[jt.arange(len(targets)), targets]

        # 计算GCE损失
        if self.q == 0:
            loss = -jt.log(probs)  # 当 q 接近 0 时，GCE 变为标准交叉熵损失
        else:
            loss = (1 - probs.pow(self.q)) / self.q
        
        # 返回损失的平均值
        return loss.mean()

# 示例用法
# logits = jt.randn(10, 5)  # 假设我们有10个样本和5个类别
# targets = jt.randint(0, 5, (10,))  # 随机生成10个真实标签

gce_loss = GeneralizedCrossEntropyLoss(q=0.8)
# loss = gce_loss(logits, targets)
# print(f"GCE Loss: {loss.item()}")