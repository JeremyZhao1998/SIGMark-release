from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention


def prepare_causal_attention_mask(
    num_frames: int, height_width: int, dtype: torch.dtype, device: torch.device, batch_size: int = None
) -> torch.Tensor:
    indices = torch.arange(1, num_frames + 1, dtype=torch.int32, device=device)
    indices_blocks = indices.repeat_interleave(height_width)
    x, y = torch.meshgrid(indices_blocks, indices_blocks, indexing="xy")
    mask = torch.where(x <= y, 0, -float("inf")).to(dtype=dtype)

    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class VideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.pad_mode = pad_mode
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(hidden_states)


class VideoResnetBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = VideoCausalConv3d(in_channels, out_channels, 3, 1, 0)

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = VideoCausalConv3d(out_channels, out_channels, 3, 1, 0)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = VideoCausalConv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.contiguous()
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual
        return hidden_states


class VideoMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "silu",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
    ) -> None:
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention
        # There is always at least one resnet
        resnets = [
            VideoResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)
            resnets.append(
                VideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                batch_size, num_channels, num_frames, height, width = hidden_states.shape
                hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3)
                attention_mask = prepare_causal_attention_mask(
                    num_frames, height * width, hidden_states.dtype, hidden_states.device, batch_size=batch_size
                )
                hidden_states = attn(hidden_states, attention_mask=attention_mask)
                hidden_states = hidden_states.unflatten(1, (num_frames, height, width)).permute(0, 4, 1, 2, 3)
            hidden_states = resnet(hidden_states)
        return hidden_states


class TemporalCodeEncoder(nn.Module):
    def __init__(
        self,
        temporal_code_size: int,
        latent_channels: int, 
        block_channels: int = 256,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True
    ) -> None:
        super().__init__()
        self.conv_in = VideoCausalConv3d(temporal_code_size, block_channels, kernel_size=3, stride=1)
        self.mid_block = VideoMidBlock3D(
            in_channels=block_channels,
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_channels,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )
        self.conv_norm_out = nn.GroupNorm(num_channels=block_channels, num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = VideoCausalConv3d(block_channels, latent_channels, kernel_size=3)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid_block(hidden_states)
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states
