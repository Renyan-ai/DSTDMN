from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from embedding import TemporalEmbedding, DynamicTemporalEmbedding

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input

class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))


    def forward(self, x):
    # x: [B, C, N, T]
        mean = x.mean(dim=1, keepdim=True) # 只对 C 做归一化
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta

class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TConv(nn.Module):
    def __init__(self, features=128, layer=4, length=12, dropout=0.1):
        super(TConv, self).__init__()
        layers = []
        kernel_size = int(length / layer + 1)
        for i in range(layer):
            self.conv = nn.Conv2d(features, features, (1, kernel_size))
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            layers += [nn.Sequential(self.conv, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.tcn(x) + x[..., -1].unsqueeze(-1)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.q = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

        self.memory = nn.Parameter(torch.randn(head, seq_length, num_nodes, self.d_k))
        nn.init.xavier_uniform_(self.memory)

        self.weight = nn.Parameter(torch.ones(d_model, num_nodes, seq_length))
        self.bias = nn.Parameter(torch.zeros(d_model, num_nodes, seq_length))

        apt_size = 10
        nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
        self.nodevec1, self.nodevec2 = [
            nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs
        ]

    def forward(self, input, adj_list=None):
        query, value = self.q(input), self.v(input)
        query = query.view(
            query.shape[0], -1, self.d_k, query.shape[2], self.seq_length
        ).permute(0, 1, 4, 3, 2)
        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], self.seq_length
        ).permute(
            0, 1, 4, 3, 2
        )  #B,T,H,N,D

        mom = torch.softmax(self.memory / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)
        Aapt = torch.softmax(
            F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=-1
        )
        mv = torch.einsum("hlnx, bhlny->bhlxy", mom, value)#BTHDD
        attn_qmv = torch.einsum("bhlnx, bhlxy->bhlny", query, mv)

        attn_dyn = torch.einsum("nm,bhlnc->bhlnc", Aapt, value)

        x = attn_qmv + attn_dyn
        x = (
            x.permute(0, 1, 4, 3, 2)
            .contiguous()
            .view(x.shape[0], self.d_model, self.num_nodes, self.seq_length)
        )
        x = self.concat(x)
        if self.num_nodes not in [170, 358,5]:
            x = x * self.weight + self.bias + x
        return x, self.weight, self.bias

class Encoder(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.attention = SpatialAttention(
            device, d_model, head, num_nodes, seq_length=seq_length, dropout=dropout
        )
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input, adj_list=None):
        # 64 64 170 12
        x, weight, bias = self.attention(input)
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x

class MemoryTemporalBlock(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=12, dropout=0.1,
                 M_long=8, causal=True):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.head = head
        self.d_k = d_model // head
        self.seq_length = seq_length
        self.M_long = M_long
        self.causal = causal  # 新增：控制是否使用因果遮罩

        # === QKV ===
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # === Memory (Shared across B and N) ===
        self.long_memory_k = nn.Parameter(torch.randn(head, M_long, self.d_k))
        self.long_memory_v = nn.Parameter(torch.randn(head, M_long, self.d_k))

        nn.init.xavier_uniform_(self.long_memory_k)
        nn.init.xavier_uniform_(self.long_memory_v)

        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.alpha_l = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        x: [B, N, T, D]
        """
        B, N, T, D = x.shape
        H, d_k = self.head, self.d_k

        # [B, N, T, D] -> [B, N, T, H, dk] -> [B, H, N, T, dk]
        # 这里的 view 和 permute 是低成本操作
        Q = self.W_q(x).view(B, N, T, H, d_k).permute(0, 3, 1, 2, 4)
        K = self.W_k(x).view(B, N, T, H, d_k).permute(0, 3, 1, 2, 4)
        V = self.W_v(x).view(B, N, T, H, d_k).permute(0, 3, 1, 2, 4)

        # ==========================================
        # 1. Self-Attention (Time)
        # ==========================================
        # [B,H,N,T,dk] @ [B,H,N,dk,T] -> [B,H,N,T,T]
        attn_time = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)

        # 关键修复：添加 Causal Mask (Masked Multi-Head Attention)
        if self.causal:
            # 生成上三角掩码 (Upper Triangular Mask)
            mask = torch.triu(torch.ones(T, T, device=self.device), diagonal=1).bool()
            # 将 mask 为 True 的位置填充为 -inf
            attn_time.masked_fill_(mask, -1e9)

        attn_time = F.softmax(attn_time, dim=-1)
        out_time = torch.matmul(attn_time, V)  # [B,H,N,T,dk]

        # ==========================================
        # 2. Memory Attention (Global Static)
        # ==========================================
        # 优化：避免 expand，使用 Einsum 或广播
        # Q: [B, H, N, T, dk]
        # Mem_K: [H, M, dk] -> need to align dimensions

        # 方案：先将 Q 变换为 [B*N, H, T, dk] 以匹配 broadcast
        # 或者直接利用广播:
        # Q:       [B, H, N, T, dk]
        # Mem_K_t: [1, H, 1, dk, M] (调整形状以广播)

        mem_k_t = self.long_memory_k.transpose(-1, -2).unsqueeze(0).unsqueeze(
            2)  # [1, H, 1, dk, M]

        # [B, H, N, T, dk] @ [1, H, 1, dk, M] -> [B, H, N, T, M]
        attn_long = torch.matmul(Q, mem_k_t) / math.sqrt(d_k)
        attn_long = F.softmax(attn_long, dim=-1)

        # Mem_V: [1, H, 1, M, dk]
        mem_v = self.long_memory_v.unsqueeze(0).unsqueeze(2)

        # [B, H, N, T, M] @ [1, H, 1, M, dk] -> [B, H, N, T, dk]
        out_long = torch.matmul(attn_long, mem_v)

        # ==========================================
        # 3. Fusion
        # ==========================================
        w_l = torch.sigmoid(self.alpha_l)  # ok
        out = out_time * (1 - w_l) + out_long * w_l

        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, N, T, D)

        # 这里的 norm 位置取决于你喜欢 Pre-Norm 还是 Post-Norm
        # 原代码是 Post-Norm (ResNet style)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.norm(out + x)

        return out


class TemEncoder(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=12, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.attention = MemoryTemporalBlock(
            device, d_model, head, num_nodes, seq_length=seq_length, dropout=dropout, M_long=8
        )
        #self.LayerNorm = TemporalLayerNorm(
         #   [d_model, num_nodes, seq_length],
          #  elementwise_affine=False)
        self.LayerNorm = ChannelLayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input, adj_list=None):
        # input: [B, N, T, D]

        x = self.attention(input)  # [B, N, T, D]
        x = x + input  # [B, N, T, D]

        # 调整维度以适应LayerNorm和GLU
        x = x.permute(0, 3, 1, 2)  # [B, N, T, D] -> [B, D, N, T]

        # 使用已有的LayerNorm（不是TemporalLayerNorm）
        x = self.LayerNorm(x)  # 使用self.LayerNorm
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = self.LayerNorm(x)  # 使用self.LayerNorm
        x = self.dropout2(x)

        # 变换回原始维度
        x = x.permute(0, 2, 3, 1)  # [B, D, N, T] -> [B, N, T, D]

        return x

class DSTDMN(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1

        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358  or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes>200:
            time = 96

        self.Temb = TemporalEmbedding(time, channels)
        self.T_time = DynamicTemporalEmbedding(time, channels)

        self.tconv = TConv(channels, layer=4, length=self.input_len)

        self.start_conv = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))

        self.network_channel = channels * 2

        self.SpatialBlock = Encoder(
            device,
            d_model=self.network_channel,
            head=self.head,
            num_nodes=num_nodes,
            seq_length=1,
        )

        self.TemporalBlock = TemEncoder(
            device,
            d_model=channels,
            head=4,
            num_nodes=num_nodes,
            seq_length=12,
            dropout=dropout,
        )

        self.time_proj = nn.Conv2d(
            channels, 1, kernel_size=(1, 1))

        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )

        # 门控权重生成
        self.gate = nn.Sequential(
            nn.Conv2d(output_len * 2, output_len, (1, 1)),
            nn.Sigmoid()
        )

        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        """
        :param history_data: [B, 3, N, T]
        :return:
        """
        B, D, N, T = history_data.shape
        input_data = history_data
        history_data = history_data.permute(0, 3, 2, 1) #[B, T, N, 3]
        input_data = self.start_conv(input_data) #[B, 64, N, T]

        input_data_s = self.tconv(input_data) #[B, 64, N, 1]

        x_t = input_data

        tem_emb = self.Temb(history_data)
        tem_emb_t = self.T_time(history_data)  # [B, 64, N, T]

        data_st = torch.cat([input_data_s] + [tem_emb], dim=1)

        data_tt = x_t + tem_emb_t #[B, 64, N, T]

        #spatial_out = self.SpatialBlock(data_st)
        data_st = self.SpatialBlock(data_st) + self.fc_st(data_st)# [B, 128, N, 1]
        data_st = self.regression_layer(data_st)  # [B, output_len, N, 1]

        # 添加时间注意力
        temporal_input = data_tt.permute(0, 2, 3, 1)  # [B, N, T, D]
        temporal_out = self.TemporalBlock(temporal_input)
        temporal_out = temporal_out.permute(0, 3, 1, 2)  # [B, D, N, T]
        temporal_out = self.time_proj(temporal_out) # [B, 1, N, 12]
        data_tt = temporal_out.permute(0, 3, 2, 1)

        combined = torch.cat([data_st, data_tt], dim=1)
        gate_weights = self.gate(combined)
        prediction = gate_weights * data_st + (
              1 - gate_weights) * data_tt

        return prediction


