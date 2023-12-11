import torch
import torch.nn as nn
import torch.nn.functional as F


class Proto(nn.Module):

    def __init__(self, dim, num_classes, top_k=256, gamma=0.999):
        super(Proto, self).__init__()

        self.dim = dim
        self.num_classes = num_classes

        self.prototypes = nn.Parameter(torch.randn(num_classes, dim), requires_grad=False)
        self.gamma = gamma
        self.top_k = top_k

        self.update_count = torch.zeros(num_classes)

    def update(self, features, labels, weight=None):

        self.prototypes = self.prototypes.to(features.device)
        
        features = F.interpolate(features, size=labels.shape[-2:], mode='bilinear', align_corners=False)

        if weight is not None:
            weight = F.interpolate(weight, size=labels.shape[-2:], mode='bilinear', align_corners=False)


        features = features.permute(0, 2, 3, 1).reshape(-1, self.dim).detach()
        if weight is not None:
            weight = weight.permute(0, 2, 3, 1).reshape(-1, self.num_classes).detach()
        labels = labels.reshape(-1)

        mask = labels != 255
        features = features[mask]
        if weight is not None:
            weight = weight[mask]
        labels = labels[mask]

        for i in labels.unique():

            if self.update_count[i] == 0:
                gamma = 0
            else:
                gamma = min(1 - 1 / (self.update_count[i] + 1), self.gamma)

            mask_i = labels == i
            if mask_i.sum() == 0:
                continue
            feat_i = features[mask_i]
            if weight is not None:
                w_i = weight[:, i][mask_i]
                pos = torch.argsort(w_i, descending=True)[:self.top_k]
                feat_i = feat_i[pos]
                w_i = w_i[pos].unsqueeze(0)
                if w_i.sum() == 0:
                    continue
                proto = (w_i @ feat_i) / w_i.sum()
            else:
                proto = feat_i.mean(dim=0, keepdim=True)
            proto = F.normalize(proto, dim=1)
            self.prototypes[i] = gamma * self.prototypes[i] + (1 - gamma) * proto
            
            self.update_count[i] += 1
        
        self.prototypes.data = F.normalize(self.prototypes, dim=1)

