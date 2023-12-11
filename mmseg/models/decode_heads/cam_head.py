import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from ..builder import HEADS
from .decode_head import BaseDecodeHead

@HEADS.register_module()
class CAMHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        self.loss_weight = kwargs.pop('loss_weight', 1.0)
        super(CAMHead, self).__init__(**kwargs)
        del self.conv_seg
        self.conv_cam = nn.ModuleList()
        if isinstance(self.in_channels, int):
            self.conv_cam.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)))
        else:
            for i in range(len(self.in_index)):
                self.conv_cam.append(nn.Sequential(
                    nn.Conv2d(self.in_channels[i], self.num_classes, kernel_size=1)))

        self.loss_decode = nn.BCEWithLogitsLoss()
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        if not isinstance(x, list):
            x = [x]
        cam_maps = []
        scores = []
        for x_, conv in zip(x, self.conv_cam):
            m = conv(x_) #.detach())
            cam_maps.append(self.relu(m))
            scores.append(F.adaptive_avg_pool2d(m, 1))
        return cam_maps, scores
    
    def forward_train_step(self, 
                      inputs, 
                      img_metas, 
                      gt_semantic_seg):
        _, scores = self.forward(inputs)
        losses = self.losses(scores, gt_semantic_seg)
        return losses
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        return dict()
    
    @force_fp32(apply_to=('cam_map', ))
    def losses(self, cls_scores, seg_label):
        loss = dict()
        loss['loss_cam'] = 0
        for i, score in enumerate(cls_scores):
            cls_label = self._get_cls_label(score, seg_label)

            loss['loss_cam'] += self.loss_decode(score, cls_label) * self.loss_weight / len(cls_scores)
        return loss
    
    def _get_cls_label(self, cls_scores, seg_label, ignore_index=255):
        batch_size = seg_label.shape[0]

        cls_label = torch.zeros_like(cls_scores)
        for i in range(batch_size):
            tmp = seg_label[i].unique(sorted=True)
            if tmp[-1] == ignore_index:
                tmp = tmp[:-1]
            cls_label[i, tmp] = 1
        return cls_label

        
