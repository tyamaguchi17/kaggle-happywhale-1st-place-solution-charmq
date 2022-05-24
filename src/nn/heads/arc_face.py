import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ChannelWiseGeM(nn.Module):
    def __init__(self, dim, p=2, eps=1e-6, requires_grad=False):
        super().__init__()
        self.ps = nn.Parameter(torch.ones(dim) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        n, C, H, W = x.shape
        batch_input = x.transpose(1, 3).reshape(n * H * W, C)
        hid = batch_input.clamp(min=self.eps).pow(self.ps)
        pooled = hid.reshape(n, H * W, C).mean(1)
        return pooled.pow(1.0 / self.ps)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.30,
        easy_margin=False,
        use_penalty=True,
        initialization="xavier",
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        if initialization == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif initialization == "uniform":
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
        else:
            assert 0 == 1

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.use_penalty = use_penalty

    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if label is None:
            assert not self.use_penalty
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if not self.use_penalty:
            return self.s * cosine
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
            (1.0 - one_hot) * cosine
        )  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output, cosine


class ArcAdaptiveMarginProduct(nn.modules.Module):
    def __init__(
        self, in_features, out_features, margins, s=30.0, k=1, initialization="xavier"
    ):
        super(ArcAdaptiveMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.margins = margins
        self.k = k
        self.weight = Parameter(torch.FloatTensor(out_features * k, in_features))
        if initialization == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif initialization == "uniform":
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
        else:
            assert 0 == 1

    def forward(self, input, labels):
        # subcenter
        cosine_all = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)

        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_features).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s

        return output, cosine
