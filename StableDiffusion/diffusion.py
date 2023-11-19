import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention


class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet =Unet()
        self.final = Unet_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)
        return output

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, n_embd * 4)
        self.linear_2 = nn.Linear(n_embd * 4, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x
class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, Unet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, Unet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class Unet(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.encoder = nn.Module([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(Unet_ResidualBlock(320, 320), Unet_AttentionBlock(8, 40)),
            SwitchSequential(Unet_ResidualBlock(320, 320), Unet_AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(Unet_ResidualBlock(320, 640), Unet_AttentionBlock(8, 80)),
            SwitchSequential(Unet_ResidualBlock(640, 640), Unet_AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(Unet_ResidualBlock(640, 1280), Unet_AttentionBlock(8, 160)),
            SwitchSequential(Unet_ResidualBlock(1280, 1280), Unet_AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(Unet_ResidualBlock(1280, 1280)),
            SwitchSequential(Unet_ResidualBlock(1280, 1280)),
        ])
        self.bottle_neck = SwitchSequential(
            Unet_ResidualBlock(1280, 1280),
            Unet_AttentionBlock(8, 160),
            Unet_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.Module([
            SwitchSequential(Unet_ResidualBlock(2560, 1280)),
            SwitchSequential(Unet_ResidualBlock(2560, 1280)),
            SwitchSequential(Unet_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(Unet_ResidualBlock(2560, 1280), Unet_AttentionBlock(8, 160)),
            SwitchSequential(Unet_ResidualBlock(1920, 1280), Unet_AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(Unet_ResidualBlock(1920, 640), Unet_AttentionBlock(8, 80)),
            SwitchSequential(Unet_ResidualBlock(1280, 640), Unet_AttentionBlock(8, 80)),
            SwitchSequential(Unet_ResidualBlock(960, 640), Unet_AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(Unet_ResidualBlock(960, 640), Unet_AttentionBlock(8, 40)),
            SwitchSequential(Unet_ResidualBlock(640, 320), Unet_AttentionBlock(8, 40)),
            SwitchSequential(Unet_ResidualBlock(640, 320), Unet_AttentionBlock(8, 40)),

        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x

class Unet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Unet_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_times=1280):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_times, out_channels)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature
        feature = self.groupnorm_1(feature)
        feature = F.silu(feature)
        feature = self.conv_1(feature)
        feature = F.silu(feature)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_2(merged)
        merged = F.silu(merged)
        merged = self.conv_2(merged)
        merged = F.silu(merged)
        return merged + self.residual_layer(residue)

class Unet_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, n_embd: int, d_context = 768):
        super().__init__()
        channels = n_heads * n_embd
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)


    def forward(self, x, context):
        residue_long = x
        n, c, h, w = x.shape

        x = x.view(n, c, h*w)
        x = x.transpose(-1, -2)

        residue_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short

        residue_short = x

        x = self.layernorm_2(x)
        self.attention_2(x, context)
        x += residue_short

        residue_short = x

        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return self.conv_output(x) + residue_long
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)