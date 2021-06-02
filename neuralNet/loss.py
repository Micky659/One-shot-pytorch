import torch
import torch.nn.functional as func


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_dist = func.pairwise_distance(out1, out2, keepdim=True)
        loss = torch.mean((1-label)*torch.pow(euclidean_dist, 2) +
                          label*torch.pow(torch.clamp((self.margin -
                                                       euclidean_dist),
                                                      min=0.0), 2))
        return loss
