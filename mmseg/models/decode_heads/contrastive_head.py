import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.runner import force_fp32
import random

from ..builder import HEADS
from .decode_head import BaseDecodeHead

@HEADS.register_module()
class ContrastiveHead(BaseDecodeHead):

    def __init__(self, proto_dims=512, num_samples=128, tau=0.1, **kwargs):
        self.loss_weight = kwargs.pop('loss_weight', 1.0)
        self.proto_dims = proto_dims
        self.num_samples = num_samples
        self.tau = tau
        self.reg_norm = kwargs.pop('reg_norm', -114514)
        self.reg_weight = kwargs.pop('reg_weight', 1.0)
        super(ContrastiveHead, self).__init__(**kwargs)
        del self.conv_seg

    def forward(self, inputs, seg_logits, projs):
        
        inputs_t = self._transform_inputs(inputs)
        if not isinstance(inputs_t, list):
            inputs_t = [inputs_t]
        seg_logits = seg_logits.squeeze(1).clone().detach()
        ignore_idx = torch.where(seg_logits == 255)
        seg_logits[ignore_idx] = 0
        seg_logits_one_hot = F.one_hot(seg_logits, self.num_classes)
        seg_logits_one_hot[ignore_idx] = 0
        seg_logits_one_hot = seg_logits_one_hot.permute(0, 3, 1, 2).float()

        vs, labels = [], []
        for x, proj in zip(inputs_t, projs):
            v = proj(x)

            b, d, h, w = v.shape
            label = F.interpolate(
                input=seg_logits_one_hot,
                size=(h, w),
                mode='area').permute(0, 2, 3, 1)
            v = v.permute(0, 2, 3, 1).contiguous().view(-1, self.proto_dims)
            label = label.view(-1, self.num_classes)

            v = F.normalize(v, p=2, dim=1)

            vs.append(v)
            labels.append(label)

        return vs, labels
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        return dict()


    def forward_train_step(self, inputs, prototypes, seg_logits, proj):
        vs, labels = self.forward(inputs, seg_logits, proj)
        losses = self.losses(vs, labels, prototypes)
        return losses

    def contrastive_loss(self, vs, labels, prototypess):

        loss = dict()
        loss['loss_cl'] = 0
        for i, (v, label, prototypes) in enumerate(zip(vs, labels, prototypess)):
            indices = random.sample(range(v.shape[0]), int(self.num_samples * v.shape[0])) # torch.randperm(v.shape[0])[:int(self.num_samples * v.shape[0])]
            v_samples = v[indices]
            lbl_samples = label[indices]

            loss[f'loss_cl'] += nn.CrossEntropyLoss()((v_samples @ prototypes.T) / self.tau, lbl_samples) * self.loss_weight / len(vs)

        return loss
    
    def reg_loss(self, vs, prototypess):

        loss = dict()
        loss['reg_loss'] = 0
        for i, (v, prototypes) in enumerate(zip(vs, prototypess)):
            v_mean = v.mean(axis=0, keepdim=True)
            logits = v_mean.mm(prototypes.detach().permute(1, 0)) / \
                self.tau
            loss[f'reg_loss'] += \
                torch.sum(torch.softmax(logits, dim=1).log()) / self.reg_norm * self.reg_weight / len(vs)
        
        return loss

    @force_fp32(apply_to=('v_samples', ))
    def losses(self, v_samples, lbl_samples, prototypes):
        loss = dict()
        loss.update(self.contrastive_loss(v_samples, lbl_samples, prototypes))
        if self.reg_norm > 0:
            loss.update(self.reg_loss(v_samples, prototypes))
        return loss
        
            




