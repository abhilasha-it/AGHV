"""AGHV-Net: Attention-Guided Hybrid Vision Network.

A hybrid CNN + Vision Transformer architecture for flower classification.

Design:
  1. CNN branch (ResNet-50 backbone) extracts local, texture-sensitive
     spatial features -> a (B, C_cnn, H, W) feature map.
  2. ViT branch (timm ViT) extracts globally contextualized patch tokens
     -> a (B, N, D_vit) sequence.
  3. Attention-Guided Fusion: the CNN feature map is projected into tokens
     and fused with the ViT tokens via bidirectional multi-head
     cross-attention, so each branch's representation is refined by the
     other before pooling and classification.
  4. Auxiliary CNN-only and ViT-only classifier heads are trained via deep
     supervision alongside the fused head. Their softmax confidences are
     exactly the "CNN conf." / "ViT conf." signals consumed by the ANFIS
     fusion controller.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CNNBranch(nn.Module):
    """ResNet-50 truncated before global pooling; outputs a spatial feature map."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, H/32, W/32) -> 7x7 for 224 input
        return x


class ViTBranch(nn.Module):
    """timm Vision Transformer used purely as a patch-token feature extractor."""

    def __init__(self, model_name: str = "vit_small_patch16_224", pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim
        self.num_prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(x)  # (B, 1 + N, D) including CLS/prefix tokens
        return tokens[:, self.num_prefix_tokens:, :]  # drop CLS -> (B, N, D)


class CrossAttentionBlock(nn.Module):
    """Query sequence attends over a key/value sequence, with a residual MLP."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, attn_weights = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        query = query + attn_out
        query = query + self.mlp(self.norm_mlp(query))
        return query, attn_weights


class AttentionGuidedFusion(nn.Module):
    """Projects CNN spatial features into tokens and bidirectionally cross-attends
    them with ViT tokens, so each branch's representation is guided by the other.
    """

    def __init__(self, cnn_channels: int, vit_dim: int, fusion_dim: int = 384, num_heads: int = 6):
        super().__init__()
        self.cnn_proj = nn.Conv2d(cnn_channels, fusion_dim, kernel_size=1)
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)

        self.cnn_guided_by_vit = CrossAttentionBlock(fusion_dim, num_heads)
        self.vit_guided_by_cnn = CrossAttentionBlock(fusion_dim, num_heads)

        self.fusion_dim = fusion_dim

    def forward(self, cnn_map: torch.Tensor, vit_tokens: torch.Tensor):
        b, c, h, w = cnn_map.shape
        cnn_tokens = self.cnn_proj(cnn_map).flatten(2).transpose(1, 2)  # (B, H*W, fusion_dim)
        vit_tokens = self.vit_proj(vit_tokens)  # (B, N, fusion_dim)

        cnn_refined, cnn_attn = self.cnn_guided_by_vit(cnn_tokens, vit_tokens)
        vit_refined, vit_attn = self.vit_guided_by_cnn(vit_tokens, cnn_tokens)

        cnn_pooled = cnn_refined.mean(dim=1)
        vit_pooled = vit_refined.mean(dim=1)
        fused = torch.cat([cnn_pooled, vit_pooled], dim=1)  # (B, 2 * fusion_dim)

        attention_maps = {"cnn_guided_by_vit": cnn_attn, "vit_guided_by_cnn": vit_attn}
        return fused, cnn_pooled, vit_pooled, attention_maps


class AGHVNet(nn.Module):
    """Attention-Guided Hybrid Vision Network for flower classification.

    Returns logits from the fused head plus auxiliary CNN-only / ViT-only
    logits (used for deep supervision and as ANFIS confidence inputs).
    """

    def __init__(self, num_classes: int = 102, vit_model_name: str = "vit_small_patch16_224",
                 fusion_dim: int = 384, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.cnn_branch = CNNBranch(pretrained=pretrained)
        self.vit_branch = ViTBranch(vit_model_name, pretrained=pretrained)
        self.fusion = AttentionGuidedFusion(self.cnn_branch.out_channels, self.vit_branch.embed_dim, fusion_dim)

        self.fused_head = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, num_classes),
        )
        self.cnn_aux_head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, num_classes))
        self.vit_aux_head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, num_classes))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        cnn_map = self.cnn_branch(x)
        vit_tokens = self.vit_branch(x)
        fused, cnn_pooled, vit_pooled, attn_maps = self.fusion(cnn_map, vit_tokens)

        logits = self.fused_head(fused)
        cnn_logits = self.cnn_aux_head(cnn_pooled)
        vit_logits = self.vit_aux_head(vit_pooled)

        out = {"logits": logits, "cnn_logits": cnn_logits, "vit_logits": vit_logits}
        if return_attention:
            out["attention_maps"] = attn_maps
        return out

    @torch.no_grad()
    def predict_with_confidences(self, x: torch.Tensor):
        """Convenience for the ANFIS simulator: returns predicted class, fused
        confidence, CNN-branch confidence, and ViT-branch confidence for a batch.
        """
        self.eval()
        out = self.forward(x)
        fused_probs = F.softmax(out["logits"], dim=1)
        cnn_probs = F.softmax(out["cnn_logits"], dim=1)
        vit_probs = F.softmax(out["vit_logits"], dim=1)

        fused_conf, pred_class = fused_probs.max(dim=1)
        cnn_conf = cnn_probs.max(dim=1).values
        vit_conf = vit_probs.max(dim=1).values
        return {
            "pred_class": pred_class,
            "fused_conf": fused_conf,
            "cnn_conf": cnn_conf,
            "vit_conf": vit_conf,
        }


def aghv_net_loss(outputs: dict, targets: torch.Tensor, aux_weight: float = 0.3) -> torch.Tensor:
    """Deep-supervision loss: fused head + weighted auxiliary CNN/ViT heads."""
    main_loss = F.cross_entropy(outputs["logits"], targets)
    cnn_loss = F.cross_entropy(outputs["cnn_logits"], targets)
    vit_loss = F.cross_entropy(outputs["vit_logits"], targets)
    return main_loss + aux_weight * (cnn_loss + vit_loss)


if __name__ == "__main__":
    model = AGHVNet(num_classes=102, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    print({k: v.shape for k, v in outputs.items()})
