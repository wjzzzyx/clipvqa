import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if ''!=act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if ''!=act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """

    def __init__(self, v_dim, q_dim, h_dim, glimpse, act='ReLU', dropout=[.2, .5], k=3):  # 128,1024,1024,2
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.glimpse = glimpse

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == glimpse:
            pass
        elif glimpse <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, glimpse, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, glimpse, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, glimpse), dim=None)

    def forward(self, v, q):
        v = v.to(torch.float32)
        if None == self.glimpse:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = d_.transpose(1, 2).transpose(2, 3)  # b x v x q x h_dim
            return logits

        elif self.glimpse <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat  # broadcast, b x glimpse x v x h_dim
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3))  # b x glimpse x v x q
            logits = logits + self.h_bias
            return logits  # b x glimpse x v x q

        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x glimpse
            return logits.transpose(2, 3).transpose(1, 2)  # b x glimpse x v x q

    def forward_with_weights(self, v, q, w):
        v = v.to(torch.float32)
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(2)  # b x d x 1 x v
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(3)  # b x d x q x 1
        logits = torch.matmul(torch.matmul(v_.float(), w.unsqueeze(1).float()), q_.float()).type_as(v_)  # b x d x 1 x 1
        # logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)# b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2)
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits


# Bilinear Attention
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):  #128, 1024, 1024,2
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(
            BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):  # v:32,1,128; q:32,12,1024
        """
        v: [batch, 1, vdim]
        q: [batch, seq, qdim]
        """
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # batch x glimpse x v_seq x q_seq

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = F.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits


class BiResNet(nn.Module):
    def __init__(self, cfg, dataset, priotize_using_counter=False):
        super(BiResNet,self).__init__()
        # Optional module: counter
        use_counter = cfg.TRAIN.ATTENTION.USE_COUNTER if priotize_using_counter is None else priotize_using_counter
        if use_counter or priotize_using_counter:
            objects = 10  # minimum number of boxes
        if use_counter or priotize_using_counter:
            counter = Counter(objects)
        else:
            counter = None
        # # init Bilinear residual network
        b_net = []   # bilinear connect :  (XTU)T A (YTV)
        q_prj = []   # output of bilinear connect + original question-> new question    Wq_ +q
        c_prj = []
        for i in range(cfg.TRAIN.ATTENTION.GLIMPSE):
            b_net.append(BCNet(cfg.TRAIN.VISION.POOL_DIM, cfg.TRAIN.QUESTION.HID_DIM, cfg.TRAIN.QUESTION.HID_DIM, None, k=1))
            q_prj.append(FCNet([cfg.TRAIN.QUESTION.HID_DIM, cfg.TRAIN.QUESTION.HID_DIM], '', .2))
            if use_counter or priotize_using_counter:
                c_prj.append(FCNet([objects + 1, cfg.TRAIN.QUESTION.HID_DIM], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.cfg = cfg

    def forward(self, v_emb, q_emb, att_p):
        b_emb = [0] * self.cfg.TRAIN.ATTENTION.GLIMPSE
        for g in range(self.cfg.TRAIN.ATTENTION.GLIMPSE):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att_p[:,g,:,:]) # b x l x h
            # atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        return q_emb.sum(1)