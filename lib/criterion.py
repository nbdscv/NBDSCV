import torch
import torch.nn.functional as F
import torch.nn as nn


class UELoss(nn.Module):
    def __init__(self):
        super(UELoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
        

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.diff = nn.MSELoss()

    def forward(self, x, y, structure):
        batchSize = x.size(0)
        preds = F.softmax(x, 1)
        neighbor_indexes_sim = structure.neighbor_indexes_sim[y]
        neighbor_indexes_disim = structure.neighbor_indexes_disim[y]

        x_inst = preds.gather(1, neighbor_indexes_sim[:,0].view(-1,1)).sum(1)
        l_inst = -1 * torch.log(x_inst).sum(0)
        
        l_sim = 0
        if neighbor_indexes_sim.size(1) > 1:
            x_sim = preds.gather(1, neighbor_indexes_sim).sum(1)
            l_sim = -1 * torch.log(x_sim).sum(0)

        l_disim = 0
        if neighbor_indexes_disim.size(1) > 1:
            x_disim = preds.gather(1, neighbor_indexes_disim).sum(1)
            l_disim = -1 * torch.log(1-x_disim).sum(0)

        NBHD_loss = (l_inst + l_sim + l_disim) / batchSize
        return NBHD_loss

