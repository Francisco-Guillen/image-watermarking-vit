import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def interpolate_pos_embed(vit: nn.Module, img_size: int):
    """Interpolate ViT pos_embed (224 -> 256 etc.). Works for timm ViT with cls token."""
    if not hasattr(vit, "pos_embed") or vit.pos_embed is None:
        return

    pos = vit.pos_embed  # (1, 1+N, D)
    num_extra = 1  # cls token
    n = pos.shape[1] - num_extra
    gs_old = int(n ** 0.5)

    patch = vit.patch_embed.patch_size[0]
    gs_new = img_size // patch

    if gs_new == gs_old:
        return

    extra = pos[:, :num_extra]          # (1,1,D)
    grid = pos[:, num_extra:]           # (1,N,D)
    dim = grid.shape[-1]

    grid = grid.reshape(1, gs_old, gs_old, dim).permute(0, 3, 1, 2)  # 1,D,H,W
    grid = F.interpolate(grid, size=(gs_new, gs_new), mode="bilinear", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, dim)

    vit.pos_embed = nn.Parameter(torch.cat([extra, grid], dim=1))


class ViTSecretDecoder(nn.Module):
    """
    Drop-in secret decoder:
      input:  (B,3,256,256)
      output: (B,msg_len) logits
    """
    def __init__(self, msg_len=100, img_size=256, token_drop=0.0, pretrained=True):
        super().__init__()
        self.msg_len = msg_len
        self.img_size = img_size
        self.token_drop = float(token_drop)

        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            img_size=img_size,   # <- isto é o crucial
            num_classes=0,
            global_pool="",
        )
        interpolate_pos_embed(self.vit, img_size=img_size)

        self.head_norm = nn.LayerNorm(self.vit.num_features)
        self.head = nn.Linear(self.vit.num_features, msg_len)

    def forward(self, x):
        # ---- preprocess for pretrained ViT (expects ImageNet-normalized [0,1]) ----
        # your pipeline produces ~[-1,1]
        x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)

        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std

        # ---- patch tokens ----
        t = self.vit.patch_embed(x)  # (B, N, D)
        cls = self.vit.cls_token.expand(t.shape[0], -1, -1)  # (B,1,D)
        t = torch.cat([cls, t], dim=1)  # (B,1+N,D)

        # ---- pos + dropout ----
        t = t + self.vit.pos_embed
        t = self.vit.pos_drop(t)

        # ---- token dropout (do not drop CLS) ----
        if self.training and self.token_drop > 0:
            B, T, D = t.shape
            n = T - 1
            keep = max(1, int(n * (1.0 - self.token_drop)))

            idx = torch.rand(B, n, device=t.device).argsort(dim=1)[:, :keep] + 1
            cls_tok = t[:, :1, :]
            kept = t.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
            t = torch.cat([cls_tok, kept], dim=1)

        # ---- transformer ----
        for blk in self.vit.blocks:
            t = blk(t)
        t = self.vit.norm(t)

        # ---- mean pooling over patch tokens (ignore CLS) ----
        feat = t[:, 1:, :].mean(dim=1)
        feat = self.head_norm(feat)
        logits = self.head(feat)
        return logits



