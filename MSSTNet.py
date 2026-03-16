


'''
Usage:


'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class NeighborConv(nn.Module):
    """
    近邻卷积：对每个channel取m邻域，在邻域维做( m, 1 )卷积压缩回1。
    等价于：对每个 (b,f,c,t) ，对邻域m个值做共享卷积核加权和。
    """

    def __init__(self, F1: int, C: int, m: int, nbr_idx: torch.Tensor):
        super().__init__()
        assert nbr_idx.shape == (C, m)
        self.C = C
        self.m = m
        self.register_buffer("nbr_idx", nbr_idx.long())  # (C,m)
        self.dw = nn.Conv2d(in_channels=F1, out_channels=F1, kernel_size=(m, 1))

    def forward(self, x):
        # x: (B,F1,C,T)
        B, F1, C, T = x.shape
        assert C == self.C

        # 取邻域：得到 (B,F1,C,m,T)
        # 用 index_select/advanced indexing 会生成临时5维，但不做Conv3d，开销可控
        x_n = x[:, :, self.nbr_idx, :]  # (B,F1,C,m,T)
        # 调整为 Conv2d 需要的 (B*C, F1, m, T)
        x_n = x_n.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,F1,m,T)
        x_n = x_n.view(B * C, F1, self.m, T)  # (B*C,F1,m,T)
        y = self.dw(x_n)
        y = y.view(B, C, F1, 1, T).squeeze(3)  # (B,C,F1,T)
        y = y.permute(0, 2, 1, 3).contiguous()  # (B,F1,C,T)
        return y


class MultiScaleNeighborConv(nn.Module):
    """
    输入:  x (B,F1,C,T)
    输出:  y (B,F1,C,T)

    多尺度：每个尺度 m 输出 out_s (B,F1,C,T)
    融合：不依赖 trial，只依赖通道/尺度的可学习权重（或 mean）

    fusion_mode:
      - "mean": 直接均值融合（你当前最稳）
      - "scale": 每个尺度一个可学习标量权重 alpha_s
      - "scale_channel": 每个(尺度,通道)一个可学习权重 alpha_{s,c}（推荐）
    """

    def __init__(self, F1, C, nbr_idx_dict, scales=(3, 5),
                 fusion_mode="scale_channel",
                 init_mode="uniform",
                 temperature=1.0,
                 use_softmax=True):
        super().__init__()
        self.scales = list(scales)
        self.S = len(self.scales)
        self.F1 = F1
        self.C = C

        assert fusion_mode in ["mean", "scale", "scale_channel"]
        self.fusion_mode = fusion_mode

        # multi-scale neighbor convs
        self.convs = nn.ModuleList([
            NeighborConv(F1, C, m, nbr_idx_dict[m]) for m in self.scales
        ])

        # temperature for softmax (if used)
        self.temperature = float(temperature)
        self.use_softmax = bool(use_softmax)

        # learnable fusion parameters (static, no dependence on B)
        if self.fusion_mode == "scale":
            # logits: (S,)
            self.alpha_logits = nn.Parameter(torch.zeros(self.S))
            if init_mode == "uniform":
                nn.init.constant_(self.alpha_logits, 0.0)  # softmax -> uniform
        elif self.fusion_mode == "scale_channel":
            # logits: (S,C)
            self.alpha_logits = nn.Parameter(torch.zeros(self.S, self.C))
            if init_mode == "uniform":
                nn.init.constant_(self.alpha_logits, 0.0)  # softmax over S -> uniform per channel

    def _get_alpha(self, device):
        """
        return alpha for fusion
        - mean: None
        - scale: (S,)
        - scale_channel: (S,C)
        """
        if self.fusion_mode == "mean":
            return None

        logits = self.alpha_logits.to(device)

        if self.use_softmax:
            # normalize across scales dimension (S)
            if self.fusion_mode == "scale":
                alpha = torch.softmax(logits / self.temperature, dim=0)  # (S,)
            else:
                alpha = torch.softmax(logits / self.temperature, dim=0)  # (S,C)  softmax over S for each C
        else:
            # alternative: sigmoid weights (no sum-to-1). usually less stable than softmax
            alpha = torch.sigmoid(logits)

        return alpha

    def forward(self, x):  # x: (B,F1,C,T)
        outs = [conv(x) for conv in self.convs]  # list length S, each (B,F1,C,T)
        H = torch.stack(outs, dim=1)  # (B,S,F1,C,T)

        if self.fusion_mode == "mean":
            return H.mean(dim=1)  # (B,F1,C,T)

        alpha = self._get_alpha(device=H.device)

        if self.fusion_mode == "scale":
            # alpha: (S,) -> broadcast to (1,S,1,1,1)
            y = (H * alpha.view(1, self.S, 1, 1, 1)).sum(dim=1)  # (B,F1,C,T)
            return y

        # fusion_mode == "scale_channel"
        # alpha: (S,C) -> broadcast to (1,S,1,C,1)
        y = (H * alpha.view(1, self.S, 1, self.C, 1)).sum(dim=1)  # (B,F1,C,T)
        return y


def build_knn_index_from_pos(pos, m: int, device="cpu"):
    """
    pos: (C,3) list/np/torch
    return nbr_idx: (C,m) long tensor
    """
    pos = torch.tensor(pos, dtype=torch.float32, device=device)  # (C,3)
    C = pos.shape[0]
    dist = torch.cdist(pos, pos, p=2)  # (C,C)
    # argsort: first is self (distance 0)
    order = dist.argsort(dim=1)
    nbr_idx = order[:, :m]
    return nbr_idx.long()  # (C,m)


class PaP_MS_SpaConv(nn.Module):
    def __init__(self, n_chan, pos, F1, scales):
        super().__init__()
        self.n_chan = n_chan
        self.F1 = F1
        # ---- temporal conv: keep (C) and (T) resolution ----
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(self.n_chan, self.n_chan * F1, kernel_size=(1, 25), groups=self.n_chan, bias=False),
            nn.BatchNorm2d(self.n_chan * F1),
        )
        # ---- build neighbor indices for each scale (offline once) ----
        nbr_idx_dict = {}
        for m in scales:
            nbr_idx_dict[m] = build_knn_index_from_pos(pos=pos, m=m, device="cpu")

        self.ms_spatial = MultiScaleNeighborConv(
            F1=F1, C=n_chan, nbr_idx_dict=nbr_idx_dict,
            scales=scales)  # ["mean", "scale", "scale_channel"]

    def forward(self, x):
        """
        x: (B,1,C,T)
        return dict: logits + alpha_scale + intermediate
        """
        B, _, C, T = x.shape
        assert C == self.n_chan
        # (B,1,C,T) -> (B,C,1,T)
        x = x.permute(0, 2, 1, 3).contiguous()
        h = self.temporal_conv(x)  # (B, F1*C, 1, T)
        h = h.view(B, C, self.F1, 1, h.shape[-1])  # (B,C,F1,1,T)
        h = h.squeeze(3)  # (B,C,F1,T)
        h = h.permute(0, 2, 1, 3).contiguous()  # (B,F1,C,T)
        h_ms = self.ms_spatial(h)
        return h_ms


class Avg_pooling(nn.Module):
    def __init__(self, k=64, s=16, time_out=4):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(1, k), stride=(1, s))
        self.adapt = nn.AdaptiveAvgPool2d((1, time_out))

    def forward(self, x):
        x = self.pool(x)
        x = self.adapt(x)
        return x



class Max_pooling(nn.Module):
    def __init__(self, k=64, s=16, time_out=4):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(1, k), stride=(1, s))
        self.adapt = nn.AdaptiveMaxPool2d((1, time_out))

    def forward(self, x):
        x = self.pool(x)
        x = self.adapt(x)
        return x

class Energy_pooling(nn.Module):
    """
    Energy pooling: mean(x^2) over time windows, then adaptive align.
    输出仍是能量（不是开方）。
    """

    def __init__(self, k=64, s=16, time_out=4):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(1, k), stride=(1, s))
        self.adapt = nn.AdaptiveAvgPool2d((1, time_out))

    def forward(self, x):
        x = x.pow(2)
        x = self.pool(x)
        x = self.adapt(x)
        return x


class RMS_pooling(nn.Module):
    """
    RMS pooling: sqrt(mean(x^2))，更贴近EEG幅值/功率表征。
    """

    def __init__(self, k=64, s=16, time_out=4, eps=1e-6):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(1, k), stride=(1, s))
        self.adapt = nn.AdaptiveAvgPool2d((1, time_out))
        self.eps = eps

    def forward(self, x):
        x = x.pow(2)
        x = self.pool(x)
        x = self.adapt(x)
        x = torch.sqrt(x.clamp_min(self.eps))
        return x


def build_temporal_pool(pool_type: str, k=64, s=16, time_out=16):
    pool_type = pool_type.lower()
    if pool_type == "avg":
        return Avg_pooling(k=k, s=s, time_out=time_out)
    if pool_type == "energy":
        return Energy_pooling(k=k, s=s, time_out=time_out)
    if pool_type == "rms":
        return RMS_pooling(k=k, s=s, time_out=time_out)
    if pool_type == "max":
        return Max_pooling(k=k, s=s, time_out=time_out)

    raise ValueError(f"Unknown pool_type={pool_type}")


class TemporalCompressor(nn.Module):
    """
    时域压缩模块：将 Time 维压缩为 1，同时进行通道扩充（可选）。
    替代了你代码中手写的 nn.Sequential(Conv1d...)
    """

    def __init__(self, in_channels, time_len, expansion=3):
        super().__init__()
        out_channels = in_channels * expansion

        self.compress = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=time_len, groups=in_channels, bias=False),
        )

    def forward(self, x):
        # x: (B, C, T) -> (B, 3*C, 1)
        return self.compress(x)


# --- 3. 主模型 ---
class MSSTNet(nn.Module):
    def __init__(self, n_chan, n_time, num_classes, para=None):
        super().__init__()
        assert para is not None and "pos" in para, "para must include 'pos'"
        self.pos = para["pos"]

        self.F1 = int(para.get("F1", 8))
        self.scales = tuple(para.get("spatial_scales", (3, 5)))

        # 总通道数 = 基础 filter * 3 (因为要拆分给3个分支)
        base_filter = int(para.get("filter", 20))
        self.num_filter = base_filter * 3

        # 每个分支实际处理的通道数
        self.branch_dim = base_filter

        self.time_out = list(para.get("time_out", (4, 12, 36)))
        self.pool_type = para.get("pooling", 'rms')

        self.n_chan = n_chan
        self.n_time = n_time
        self.nClass = num_classes
        self.min_crop_ratio = 0.6
        self.max_crop_ratio = 1.0

        print("=" * 50)
        print(f"[{self.__class__.__name__}] Hyper-parameters:")
        for key, value in para.items():
            # 特殊处理 pos，防止打印整个坐标矩阵刷屏
            if key == "pos":
                # 如果是 Tensor 或 Numpy，打印形状
                if hasattr(value, "shape"):
                    val_str = f"<Shape: {value.shape}>"
                else:
                    val_str = "<Coordinate Data>"
            else:
                val_str = str(value)

            print(f"  {key:<20} : {val_str}")
        print("=" * 40 + "\n")

        # 1. 空间卷积
        self.ms_spa_conv = PaP_MS_SpaConv(n_chan, self.pos, self.F1, self.scales)

        # 2. 时空主干 (输出 3*F 通道)
        self.st_conv = nn.Sequential(
            nn.Conv2d(self.F1, self.num_filter, kernel_size=(n_chan, 1), bias=False),
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=(1, 32), groups=self.num_filter, bias=False),
            nn.BatchNorm2d(self.num_filter),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        # 3. 多尺度池化
        self.pool_out1 = build_temporal_pool(pool_type=self.pool_type, time_out=self.time_out[0])  #
        self.pool_out2 = build_temporal_pool(pool_type=self.pool_type, time_out=self.time_out[1])  #
        self.pool_out3 = build_temporal_pool(pool_type=self.pool_type, time_out=self.time_out[2])  #

        self.proj_head_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.branch_dim * self.time_out[0], 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64)  # 最终 embedding 维度，通常 64 或 128
        )

        self.proj_head_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.branch_dim * self.time_out[1], 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64)  # 最终 embedding 维度，通常 64 或 128
        )
        self.proj_head_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.branch_dim * self.time_out[2], 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 64)  # 最终 embedding 维度，通常 64 或 128
        )


        self.head1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.time_out[0] * self.branch_dim, num_classes),
            nn.LogSoftmax(dim=-1)
        )
        self.head2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.time_out[1] * self.branch_dim, num_classes),
            nn.LogSoftmax(dim=-1)
        )
        self.head3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.time_out[2] * self.branch_dim, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        self.final_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sum(self.time_out) * (self.branch_dim), num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        """
        x: (B,1,C,T)
        """
        # 1. 骨干提取
        if self.training:
            B, _, C, T = x.shape
            ratio = random.uniform(self.min_crop_ratio, self.max_crop_ratio)
            crop_len = max(1, int(T * ratio))
            max_start = T - crop_len
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            x = x[..., start_idx:start_idx + crop_len]

        h_ms_spa = self.ms_spa_conv(x)
        feat = self.st_conv(h_ms_spa)  # (B, 3*F, 1, T_raw)
        # 2. 拆分特征
        # 沿着 dim=1 切成三个 (B, F, 1, T_raw)
        x1_raw, x2_raw, x3_raw = torch.chunk(feat, chunks=3, dim=1)

        f1 = self.pool_out1(x1_raw).squeeze(dim=2)
        f2 = self.pool_out2(x2_raw).squeeze(dim=2)
        f3 = self.pool_out3(x3_raw).squeeze(dim=2)
        feat = torch.cat([f1, f2, f3], dim=-1)

        logits1 = self.head1(f1)
        logits2 = self.head2(f2)
        logits3 = self.head3(f3)
        logits = self.final_head(feat)  # 主 logits (预测结果)

        # 【核心修改 2】计算三个分支的投影特征
        proj_feat_1 = F.normalize(self.proj_head_1(f1), dim=1)
        proj_feat_2 = F.normalize(self.proj_head_2(f2), dim=1)
        proj_feat_3 = F.normalize(self.proj_head_3(f3), dim=1)

        # 返回字典以便 Loss_func 处理
        return logits, [logits1, logits2, logits3], [proj_feat_1, proj_feat_2, proj_feat_3]



# --- 1. SupCon Loss (保持你提供的代码不变) ---
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        sim_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss


class Loss_func(nn.Module):
    def __init__(self, para):
        super().__init__()
        pos = torch.tensor(para["pos"], dtype=torch.float32)
        self.register_buffer("pos", pos)

        self.alpha_aux = para.get("lambda_aux", 0.3)
        self.alpha_supcon = para.get("lambda_cvc", 0.1)
        self.temperature_supcon = para.get("temperature", 0.5)
        self.supcon_loss = SupConLoss(temperature=self.temperature_supcon)


    def forward(self, model_output, targets):
        logits, logit_list, proj_list = model_output

        # 1. 主分类 Loss (使用 CrossEntropyLoss，自带 LogSoftmax)
        loss_main = F.nll_loss(logits, targets)


        # 2. 辅助分类 Loss
        l1 = F.nll_loss(logit_list[0], targets)
        l2 = F.nll_loss(logit_list[1], targets)
        l3 = F.nll_loss(logit_list[2], targets)
        loss_aux = (l1 + l2 + l3) / 3


        # 3. 统一多视角对比损失
        # 拼接特征 (3B, 64)
        features_all = torch.cat(proj_list, dim=0)
        batch_size = targets.shape[0]
        device = targets.device
        instance_ids = torch.arange(batch_size).to(device)
        targets_instance = torch.cat([instance_ids, instance_ids, instance_ids], dim=0)

        loss_align_cvc = self.supcon_loss(features_all, targets_instance)
        loss_align1 = self.supcon_loss(proj_list[0], targets)
        loss_align2 = self.supcon_loss(proj_list[1], targets)
        loss_align3 = self.supcon_loss(proj_list[2], targets)
        loss_align = loss_align_cvc  + (loss_align1 + loss_align2 + loss_align3)
        # print(loss_align_cvc, loss_align1, loss_align2, loss_align3)

        # 4. 总 Loss
        loss = loss_main + self.alpha_aux * loss_aux + self.alpha_supcon * loss_align

        # 返回纯分类 loss (main) 用于早停判断
        return loss, logits, loss_main






if __name__ == "__main__":
    pass