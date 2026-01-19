import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class DisAttention(nn.Module):
    def __init__(self, embed_dim, alpha, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def topk_mask(self, A, k):
        B, M, N = A.shape
        A = A.reshape(B, -1)
        _, topk_idx = torch.topk(A, k=k, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(-1, topk_idx, 1.0)
        mask = mask.reshape(B, M, N)
        return mask

    def topk_gumbel_softmax_mask(self, A, k, tau=1.0, hard=True):
        B, M, N = A.shape
        A_flat = A.reshape(B, -1)
        gumbel_noise = -torch.empty_like(A_flat).exponential_().log()
        logits = A_flat + gumbel_noise
        soft_mask = F.softmax(logits / tau, dim=-1)
        if hard:
            topk_val, topk_idx = torch.topk(A_flat, k=k, dim=-1)
            hard_mask = torch.zeros_like(A_flat)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            soft_mask = (hard_mask - soft_mask).detach() + soft_mask
        return soft_mask.reshape(B, M, N)

    def global_softmax(self, attn_scores):
        B, M, N = attn_scores.shape
        attn_flat = attn_scores.view(B, -1)
        attn_flat_soft = F.softmax(attn_flat, dim=-1)
        attn_weights = attn_flat_soft.view(B, M, N)
        return attn_weights

    def forward(self, query, key, value):
        B, Q_len, D = query.shape
        _, K_len, _ = key.shape

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        _attn_weights = F.softmax(-attn_scores, dim=-1)

        maskI = self.topk_mask(attn_weights, int(Q_len * K_len * self.alpha))
        maskV = 1 - maskI

        outI = torch.matmul(maskI * attn_weights, V)
        outV = torch.matmul(maskV * _attn_weights, V)

        return outI, outV, maskI * attn_weights


class Patch_Embedding(nn.Module):
    def __init__(self, seq_len, patch_num, patch_len, d_model, d_ff, variate_num):
        super(Patch_Embedding, self).__init__()
        self.pad_num = patch_num * patch_len - seq_len
        self.patch_len = patch_len
        self.linear = nn.Sequential(
            nn.LayerNorm([variate_num, patch_num, patch_len]),
            nn.Linear(patch_len, d_ff),
            nn.LayerNorm([variate_num, patch_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm([variate_num, patch_num, d_model]),
            nn.ReLU())

    def forward(self, x):
        x = nn.functional.pad(x, (0, self.pad_num))
        x = x.unfold(2, self.patch_len, self.patch_len)
        x = self.linear(x)
        return x


class De_Patch_Embedding(nn.Module):
    def __init__(self, pred_len, patch_num, d_model, d_ff, variate_num):
        super(De_Patch_Embedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(patch_num * d_model, d_ff),
            nn.LayerNorm([variate_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, pred_len))

    def forward(self, x):
        x = self.linear(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.ms = configs.features == 'MS'
        self.EPS = 1e-5
        patch_len = configs.patch_len
        self.patch_num = math.ceil(configs.seq_len / patch_len)
        self.alpha = nn.Parameter(torch.ones([1]))
        self.cov_embedding = nn.Conv1d(configs.dec_in - 1, configs.dec_in - 1, kernel_size=3, padding='same',
                                       groups=configs.dec_in - 1)
        self.Disattention = DisAttention(configs.d_model, configs.lamb)
        self.disentangle = nn.Linear(configs.d_model, configs.d_model)
        self.tau = configs.tau
        self.patch_cov_embedding = Patch_Embedding(configs.seq_len, self.patch_num, patch_len, configs.d_model,
                                                   configs.d_ff, configs.dec_in - 1)
        self.patch_target_embedding = Patch_Embedding(configs.seq_len, self.patch_num, patch_len, configs.d_model,
                                                      configs.d_ff, 1)

        self.head = De_Patch_Embedding(
            configs.pred_len, self.patch_num, configs.d_model, configs.d_ff, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        b, t, n = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        mean = torch.mean(x_obj, dim=-1, keepdim=True)
        std = torch.std(x_obj, dim=-1, keepdim=True)
        x_enc = (x_enc - torch.mean(x_enc, dim=-1, keepdim=True)) / (torch.std(x_enc, dim=-1, keepdim=True) + self.EPS)
        End = x_enc[:, [-1], :] if self.ms else x_enc
        Exo = self.patch_cov_embedding(x_enc[:, :-1, :]).reshape(b, (n - 1) * self.patch_num, -1)
        End = self.patch_target_embedding(End).reshape(b, self.patch_num, -1)
        InterI, InterV, attn_weights = self.Disattention(End, Exo, Exo)

        M = self.disentangle(InterI)
        Msha = F.sigmoid(M / self.tau)
        Mspe = F.sigmoid(-M / self.tau)
        RealIntraI = Msha * InterI
        RealIntraV = Mspe * InterI
        Z = F.sigmoid(self.alpha) * End + (1 - F.sigmoid(self.alpha)) * RealIntraI

        if self.training:
            enc_out1 = self.head(Z.unsqueeze(1))
            enc_out2 = self.head((Z + InterV).unsqueeze(1))
            enc_out3 = self.head((Z + RealIntraV).unsqueeze(1))
            enc_out1 = enc_out1 * std + mean
            enc_out1 = enc_out1.permute(0, 2, 1)
            enc_out2 = enc_out2 * std + mean
            enc_out2 = enc_out2.permute(0, 2, 1)
            enc_out3 = enc_out3 * std + mean
            enc_out3 = enc_out3.permute(0, 2, 1)
            return enc_out1, enc_out2, enc_out3
        else:
            enc_out1 = self.head(Z.unsqueeze(1))
            enc_out1 = enc_out1 * std + mean
            enc_out1 = enc_out1.permute(0, 2, 1)
            return enc_out1, None, None
