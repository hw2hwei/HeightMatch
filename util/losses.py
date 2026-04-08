import torch
import torch.nn as nn
import torch.nn.functional as F
        

class UnlCrossEntropyLoss(nn.Module):
    def __init__(self, conf_thresh, local_rank=None):
        super(UnlCrossEntropyLoss, self).__init__()
        self.conf_thresh = conf_thresh
        
        self.criterion_u = nn.CrossEntropyLoss(reduction='none')
        if local_rank:
            self.criterion_u = self.criterion_u.cuda(local_rank)

    def forward(self, pred, mask, conf):
        loss_u = self.criterion_u(pred, mask)
        
        # 根据置信度阈值创建有效掩码
        valid_mask = (conf >= self.conf_thresh)
        loss_u = loss_u * valid_mask

        # 归一化有效损失值
        normalization = valid_mask.sum().item() + 1e-7
        if normalization > 0:
            loss_u = loss_u.sum() / normalization
        else:
            loss_u = torch.tensor(0.0, device=pred.device)
        return loss_u