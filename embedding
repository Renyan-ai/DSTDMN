from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicTemporalEmbedding(nn.Module):
    def __init__(self, time_steps_per_day, features):
        super(DynamicTemporalEmbedding, self).__init__()
        self.time_steps_per_day = time_steps_per_day
        self.features = features

        self.time_day_emb = nn.Parameter(torch.empty(time_steps_per_day, features))
        nn.init.xavier_uniform_(self.time_day_emb)

        self.time_week_emb = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week_emb)

    def forward(self, x):
        """
        x: [B, T, N, 3]
        """
        day_index = x[..., 1].long()   # [B, T, N]
        week_index = x[..., 2].long()  # [B, T, N]

        # 取嵌入 [B, T, N, F]
        day_emb = self.time_day_emb[day_index]
        week_emb = self.time_week_emb[week_index]

        tem_emb = day_emb + week_emb  # [B, T, N, F]

        # 调整维度为 [B, F, N, T] 以匹配后续模块
        tem_emb = tem_emb.permute(0, 3, 2, 1)

        return tem_emb


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb
