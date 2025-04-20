import torch
import torch.nn as nn
import torch.nn.functional as F


# 你的SCConv代码
class SRU(nn.Module):
    def __init__(self, channels, group_num=16, gate_threshold=0.5):
        super(SRU, self).__init__()
        self.group_num = group_num
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 动态调整group_num
        group_num = min(self.group_num, x.size(1))
        while x.size(1) % group_num != 0:
            group_num -= 1
        gn = nn.GroupNorm(num_groups=group_num, num_channels=x.size(1)).to(x.device)

        gn_x = gn(x)
        reweights = self.sigmoid(gn_x)
        info_mask = reweights >= self.gate_threshold
        noninfo_mask = reweights < self.gate_threshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(self, channels, alpha=0.5, squeeze_ratio=2, group_size=2, group_kernel_size=3):
        super(CRU, self).__init__()
        self.alpha = alpha
        self.squeeze_ratio = squeeze_ratio
        self.group_size = group_size
        self.group_kernel_size = group_kernel_size
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up_channels = int(self.alpha * x.size(1))
        low_channels = x.size(1) - up_channels
        squeeze1 = nn.Conv2d(up_channels, up_channels // self.squeeze_ratio, kernel_size=1, bias=False).to(x.device)
        squeeze2 = nn.Conv2d(low_channels, low_channels // self.squeeze_ratio, kernel_size=1, bias=False).to(x.device)

        up, low = torch.split(x, [up_channels, low_channels], dim=1)
        up, low = squeeze1(up), squeeze2(low)

        # 动态调整group_size
        group_size = max(1, min(self.group_size, up.size(1)))
        while up.size(1) % group_size != 0:
            group_size -= 1

        GWC = nn.Conv2d(up.size(1), x.size(1), kernel_size=self.group_kernel_size, stride=1,
                        padding=self.group_kernel_size // 2, groups=group_size).to(x.device)
        PWC1 = nn.Conv2d(up.size(1), x.size(1), kernel_size=1, bias=False).to(x.device)
        PWC2 = nn.Conv2d(low.size(1), x.size(1) - low.size(1), kernel_size=1, bias=False).to(x.device)

        Y1 = GWC(up) + PWC1(up)
        Y2 = torch.cat([PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class SCConv(nn.Module):
    def __init__(self, channels, group_num=16, gate_threshold=0.5, alpha=0.5, squeeze_ratio=2, group_size=2,
                 group_kernel_size=3):
        super(SCConv, self).__init__()
        self.SRU = SRU(channels, group_num=group_num, gate_threshold=gate_threshold)
        self.CRU = CRU(channels, alpha=alpha, squeeze_ratio=squeeze_ratio, group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class HAT(nn.Module):
    def __init__(self, dim):
        super(HAT, self).__init__()
        self.dim = dim
        self.num_heads = 1
        self.window_size = 1
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            SCConv(dim),
            nn.ReLU(),
            SCConv(dim),
            nn.Sigmoid()
        )
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=self.num_heads)

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 基于窗口的自注意力
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)  # [H*W, B, C]
        attn_output, _ = self.self_attention(x, x, x)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)

        return attn_output


def modify_model_ConvNeXt(model):
    # 获取最后一个卷积层的输出通道数
    in_channels = None
    for layer in reversed(model.features):
        if isinstance(layer, nn.Conv2d):
            in_channels = layer.out_channels
            break
        elif isinstance(layer, nn.Sequential):
            for sublayer in reversed(layer):
                if isinstance(sublayer, nn.Conv2d):
                    in_channels = sublayer.out_channels
                    break
            if in_channels is not None:
                break

    if in_channels is None:
        raise ValueError("模型中没有找到卷积层")

    # 在模型的最后添加HAT-SCConv块
    model.features.add_module("HAT_SCConv", HAT(in_channels))

    return model


def modify_model_Swin(model):
    # 获取最后一个卷积层的输出通道数
    in_channels = None
    for layer in reversed(model.features):
        if isinstance(layer, nn.Conv2d):
            in_channels = layer.out_channels
            break
        elif isinstance(layer, nn.Sequential):
            for sublayer in reversed(layer):
                if isinstance(sublayer, nn.Conv2d):
                    in_channels = sublayer.out_channels
                    break
            if in_channels is not None:
                break

    if in_channels is None:
        raise ValueError("模型中没有找到卷积层")

    # 在模型的最后添加HAT-SCConv块
    model.features.add_module("HAT_SCConv", HAT(in_channels))

    return model