"""
Multi-Scale Progressive Super-Resolution Model with Supervised Upsampling

Architettura: Up2 → Up2 → Up1.25 (total 5x)
Supervisione: Loss a ogni stadio (2x, 4x, 5x) contro target appropriati
"""

import sys
import os
import traceback

# Add the parent directory to Python path
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, parent_dir)

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Literal, Tuple, Dict

from Super_Resolution.models_functions import (
    PatchEmbed,
    PatchUnEmbed,
    get_resi_connection,
    window_partition,
    window_reverse,
)
from Super_Resolution.mymodel.mymodel_model import WindowAttention, RSTB


class ProgressiveUpsampler(nn.Module):
    """
    Modulo di upsampling progressivo con supervisione multi-scale.

    Ogni stadio può essere un residual CNN block o un Swin block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        block_type: Literal["cnn", "swin"] = "cnn",
        window_size: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()

        self.scale_factor = scale_factor
        self.block_type = block_type

        if block_type == "cnn":
            # Residual CNN approach
            self.feature_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode="reflect"),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode="reflect"),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode="reflect"),
            )

            # Channel adjustment if needed
            if in_channels != out_channels:
                self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.skip_conv = nn.Identity()

        elif block_type == "swin":
            # Swin Transformer approach
            self.feature_conv = nn.Conv2d(
                in_channels, out_channels, 3, 1, 1, padding_mode="reflect"
            )

            # Window attention for feature enhancement
            self.window_attention = WindowAttention(
                dim=out_channels,
                window_size=(window_size, window_size),
                num_heads=num_heads,
                qkv_bias=True,
                attn_drop=0.0,
                proj_drop=0.0,
            )

            self.norm = nn.LayerNorm(out_channels)
            self.window_size = window_size

            # Channel adjustment
            if in_channels != out_channels:
                self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.skip_conv = nn.Identity()

        # Output convolution for final image generation
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                out_channels // 2, 4, 3, 1, 1, padding_mode="reflect"
            ),  # 4 channels (RGB+NIR)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            features: Processed features for next stage
            output: Image output for supervision
        """
        # Upscale input
        x_up = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
        )

        if self.block_type == "cnn":
            # CNN processing with residual connection
            features = self.feature_conv(x_up)
            features = features + self.skip_conv(x_up)  # Residual connection

        elif self.block_type == "swin":
            # Swin processing with window attention
            features = self.feature_conv(x_up)

            # Apply window attention
            B, C, H, W = features.shape

            # Ensure dimensions are compatible with window partitioning
            mod_pad_h = (self.window_size - H % self.window_size) % self.window_size
            mod_pad_w = (self.window_size - W % self.window_size) % self.window_size
            if mod_pad_h != 0 or mod_pad_w != 0:
                features = F.pad(features, (0, mod_pad_w, 0, mod_pad_h), "reflect")
                H_pad, W_pad = features.shape[2], features.shape[3]
            else:
                H_pad, W_pad = H, W

            # Convert to sequence format for attention
            feat_seq = features.flatten(2).transpose(1, 2)  # B, H*W, C
            feat_seq = self.norm(feat_seq)

            # Apply window attention
            feat_windows = window_partition(
                feat_seq.view(B, H_pad, W_pad, C), self.window_size
            )
            feat_windows = feat_windows.view(-1, self.window_size * self.window_size, C)

            # Apply attention
            attn_windows = self.window_attention(feat_windows)

            # Reverse window partition
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            feat_enhanced = window_reverse(attn_windows, self.window_size, H_pad, W_pad)

            # Remove padding if added
            if mod_pad_h != 0 or mod_pad_w != 0:
                feat_enhanced = feat_enhanced[:, :H, :W, :]

            # Back to channel format and add residual
            feat_enhanced = feat_enhanced.permute(0, 3, 1, 2).contiguous()
            features = feat_enhanced + self.skip_conv(x_up)  # Residual connection
        else:
            # Fallback case
            features = self.feature_conv(x_up)
            features = features + self.skip_conv(x_up)

        # Generate output image for supervision
        output = self.output_conv(features)

        return features, output


class MultiScaleProgressiveModel(nn.Module):
    """
    Multi-Scale Progressive Super-Resolution Model

    Architecture:
    Input → Feature Extraction → Up2 → Up2 → Up1.25 → Output
             ↓                   ↓      ↓       ↓
             Supervision at:     2x     4x      5x
    """

    def __init__(self, cfg):
        super().__init__()

        self.scale = cfg.model.scale  # Should be 5
        self.img_size = cfg.model.img_size
        self.num_in_channels = 4
        self.emb_patch_size = cfg.model.emb_patch_size
        self.emb_dim = cfg.model.embed_dim
        self.window_size = cfg.model.window_size
        num_feat = cfg.model.num_feat

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(
            self.num_in_channels, self.emb_dim, 3, 1, 1, padding_mode="reflect"
        )

        # Feature extraction backbone (reuse existing RSTB layers)
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.emb_patch_size,
            in_chans=self.emb_dim,
            embed_dim=self.emb_dim,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=self.img_size,
            patch_size=self.emb_patch_size,
            in_chans=self.emb_dim,
            embed_dim=self.emb_dim,
        )

        # Token grid calculation
        token_grid = (
            self.img_size // self.emb_patch_size,
            self.img_size // self.emb_patch_size,
        )

        # Main feature extraction layers (reduced depth for efficiency)
        self.num_layers = min(
            len(cfg.model.depths), 2
        )  # Limit to 2 layers for efficiency
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = RSTB(
                input_resolution=token_grid,
                dim=self.emb_dim,
                depth=cfg.model.depths[i],
                num_heads=cfg.model.num_heads[i],
                window_size=cfg.model.window_size,
                img_size=self.img_size,
                patch_size=self.emb_patch_size,
                resi_connection=cfg.model.resi_connection,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.emb_dim)
        self.conv_after_body = get_resi_connection(
            cfg.model.resi_connection, self.emb_dim
        )

        # Progressive upsampling stages
        # Stage 1: Up2 (LR → 2x)
        self.upsampler_1 = ProgressiveUpsampler(
            in_channels=self.emb_dim,
            out_channels=num_feat,
            scale_factor=2.0,
            block_type="cnn",  # Can be "cnn" or "swin"
            window_size=self.window_size,
        )

        # Stage 2: Up2 (2x → 4x)
        self.upsampler_2 = ProgressiveUpsampler(
            in_channels=num_feat,
            out_channels=num_feat,
            scale_factor=2.0,
            block_type="swin",  # Mix of approaches
            window_size=self.window_size,
        )

        # Stage 3: Up1.25 (4x → 5x)
        self.upsampler_3 = ProgressiveUpsampler(
            in_channels=num_feat,
            out_channels=num_feat,
            scale_factor=1.25,
            block_type="cnn",
            window_size=self.window_size,
        )

        # Scale factors for supervision
        self.supervision_scales = [2.0, 4.0, 5.0]

    def check_image_size(self, x: Tensor) -> Tensor:
        """Check and pad image size to be multiple of window size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using RSTB layers."""
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = True
    ) -> Dict[str, Tensor]:
        """
        Forward pass with progressive upsampling and multi-scale supervision.

        Args:
            x: Input tensor [B, 4, H, W]
            return_intermediates: Whether to return intermediate outputs for supervision

        Returns:
            Dictionary with keys: 'stage_1', 'stage_2', 'final' and optionally 'features'
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # Feature extraction
        x = self.conv_first(x)
        res = x
        x = self.forward_features(x)
        x = self.conv_after_body(x) + res  # Global residual connection

        outputs = {}

        # Stage 1: Up2 (LR → 2x)
        x, out_1 = self.upsampler_1(x)
        if return_intermediates:
            outputs["stage_1"] = out_1

        # Stage 2: Up2 (2x → 4x)
        x, out_2 = self.upsampler_2(x)
        if return_intermediates:
            outputs["stage_2"] = out_2

        # Stage 3: Up1.25 (4x → 5x)
        x, out_final = self.upsampler_3(x)
        outputs["final"] = out_final

        # Crop to original scale
        for key in outputs:
            outputs[key] = outputs[key][:, :, : H * self.scale, : W * self.scale]

        return outputs
