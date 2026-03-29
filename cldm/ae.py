import numpy as np
import einops
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as thf
import pytorch_lightning as pl
import torchvision
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.model import Encoder
import lpips
from kornia import color

def patchify(z: torch.Tensor, p: int) -> torch.Tensor:
    # z: (B,C,H,W) -> (B,N,D)
    B, C, H, W = z.shape
    assert H % p == 0 and W % p == 0, f"H,W must be divisible by p. Got {(H,W)} and p={p}"
    gh, gw = H // p, W // p
    x = z.view(B, C, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, gh, gw, C, p, p)
    return x.view(B, gh * gw, C * p * p)          # (B, N, D)

def unpatchify(tokens: torch.Tensor, C: int, H: int, W: int, p: int) -> torch.Tensor:
    # tokens: (B,N,D) -> (B,C,H,W)
    B, N, D = tokens.shape
    assert D == C * p * p, f"Token dim mismatch. D={D}, expected {C*p*p}"
    gh, gw = H // p, W // p
    assert N == gh * gw, f"N mismatch. N={N}, expected {gh*gw}"
    x = tokens.view(B, gh, gw, C, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, gh, p, gw, p)
    return x.view(B, C, H, W)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class SecretEncoder3(nn.Module):
    def __init__(self, secret_len, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*3),
            nn.SiLU(),
            View(-1, 3, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, 3, 3, 3, padding=1))
        )  # secret len -> ch x res x res
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None


class SecretEncoder4(nn.Module):
    """same as SecretEncoder3 but with ch as input"""
    def __init__(self, secret_len, ch=3, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*ch),
            nn.SiLU(),
            View(-1, ch, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, ch, ch, 3, padding=1))
        )  # secret len -> ch x res x res
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None
    
class SecretEncoder6(nn.Module):
    """join img emb with secret emb"""
    def __init__(self, secret_len, ch=3, base_res=16, resolution=64, emode='c3') -> None:
        super().__init__()
        assert emode in ['c3', 'c2', 'm3']
        
        if emode == 'c3':  # c3: concat c and x each has ch channels
            secret_ch = ch 
            join_ch = 2*ch
        elif emode == 'c2':  # c2: concat c (2) and x ave (1)
            secret_ch = 2
            join_ch = ch
        elif emode == 'm3':  # m3: multiply c (ch) and x (ch)
            secret_ch = ch
            join_ch = ch       
        
        # m3: multiply c (ch) and x ave (1)
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.emode = emode
        self.resolution = resolution
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*secret_ch),
            nn.SiLU(),
            View(-1, secret_ch, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
        )  # secret len -> ch x res x res
        self.join_encoder = nn.Sequential(
            conv_nd(2, join_ch, join_ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, join_ch, ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, ch, ch, 3, padding=1),
            nn.SiLU()
        )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        if self.emode == 'c3':
            x = torch.cat([x, c], dim=1)
        elif self.emode == 'c2':
            x = torch.cat([x.mean(dim=1, keepdim=True), c], dim=1)
        elif self.emode == 'm3':
            x = x * c
        dx = self.join_encoder(x)
        dx = self.out_layer(dx)
        return dx, None
        
class SecretEncoder5(nn.Module):
    """same as SecretEncoder3 but with ch as input"""
    def __init__(self, secret_len, ch=3, base_res=16, resolution=64, joint=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.joint = joint
        self.resolution = resolution
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*ch),
            nn.SiLU(),
            View(-1, ch, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
        )  # secret len -> ch x res x res
        if joint:
            self.join_encoder = nn.Sequential(
                conv_nd(2, 2*ch, 2*ch, 3, padding=1),
                nn.SiLU(),
                conv_nd(2, 2*ch, ch, 3, padding=1),
                nn.SiLU()
            )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        if self.joint:
            x = thf.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False, antialias=True)
            c = self.join_encoder(torch.cat([x, c], dim=1))
        c = self.out_layer(c)
        return c, None


class SecretEncoder2(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.encoder.conv_out = zero_module(self.encoder.conv_out)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32*32*ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2**(log_resolution-5), 2**(log_resolution-5))),  # chx16x16 -> chx256x256
            # zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        # self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        posterior = h
        return posterior
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = torch.cat([x, c], dim=1)
        z = self.encode(x)
        # z = self.out_layer(z)
        return z, None

class SecretEncoder(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32*32*ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2**(log_resolution-5), 2**(log_resolution-5))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = x + c
        posterior = self.encode(x)
        z = posterior.sample()
        z = self.out_layer(z)
        return z, posterior

class SecretEncoderPatch(nn.Module):
    """
    Patch-wise secret injection in LATENT space.
    Input x: latent (B,3,64,64)
    Input c: secret bits (B, secret_len)
    Output eps: (B,3,64,64) to be added to latent.
    """
    def __init__(self, secret_len, patch_size=8, hidden_mult=2, alpha_max=1.0, resolution=64, ch=3) -> None:
        super().__init__()
        assert resolution % patch_size == 0, "resolution must be divisible by patch_size"
        self.secret_len = secret_len
        self.p = int(patch_size)
        self.resolution = int(resolution)
        self.ch = int(ch)
        self.alpha_max = float(alpha_max)
        self.alpha = 0.0   # vai ser actualizado no ControlAE

        patch_dim = self.ch * self.p * self.p
        hidden = patch_dim * int(hidden_mult)

        # Secret embedding -> same dim as patch token
        self.secret_embed = nn.Sequential(
            nn.Linear(secret_len, hidden),
            nn.SiLU(),
            nn.Linear(hidden, patch_dim),
        )

        # Patch modulator
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, patch_dim),
            nn.SiLU(),
            zero_module(nn.Linear(patch_dim, patch_dim)),  # start near-zero like the other encoders
        )

    def copy_encoder_weight(self, ae_model):
        return None

    def forward(self, x, c):
        # x: (B,3,64,64) latent
        # c: (B, secret_len)
        B, C, H, W = x.shape
        assert C == self.ch and H == self.resolution and W == self.resolution, \
            f"Expected latent (B,{self.ch},{self.resolution},{self.resolution}), got {tuple(x.shape)}"

        tokens = patchify(x, self.p)                 # (B, N, D)
        s = self.secret_embed(c).unsqueeze(1)        # (B, 1, D)
        s = s.expand(B, tokens.size(1), -1)          # (B, N, D)

        inp = torch.cat([tokens, s], dim=-1)         # (B, N, 2D)
        delta = self.mlp(inp)                        # (B, N, D)

        tokens_w = tokens + self.alpha * delta
        x_w = unpatchify(tokens_w, self.ch, self.resolution, self.resolution, self.p)

        eps = x_w - x                                 # return offset (eps) to match ControlAE
        if not hasattr(self, "_printed"):
            print("Patch-wise tokens:", tokens.shape,
                  "eps:", eps.shape,
                  "eps_abs_mean:", eps.abs().mean().item())
            self._printed = True

        return eps, None

class ControlAE(pl.LightningModule):
    def __init__(self,
                 first_stage_key,
                 first_stage_config,
                 control_key,
                 control_config,
                 decoder_config,
                 loss_config,
                 noise_config='__none__',
                 use_ema=False,
                 secret_warmup=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 vit_freeze_steps=2000,
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.control_key = control_key
        self.first_stage_key = first_stage_key
        self.ae = instantiate_from_config(first_stage_config)
        self.control = instantiate_from_config(control_config)
        self.decoder = instantiate_from_config(decoder_config)

        self.vit_freeze_steps = vit_freeze_steps
        self._vit_unfrozen = False

        self._is_vit = hasattr(self.decoder, "arch") and self.decoder.arch in ["vit", "vit_small", "vit_small_patch16"]
        self._has_vit_backbone = (
            self._is_vit
            and hasattr(self.decoder, "decoder")
            and hasattr(self.decoder.decoder, "vit")
        )

        if not self._has_vit_backbone:
            self.vit_freeze_steps = 0
            self._vit_unfrozen = True

        if self._has_vit_backbone:
            for p in self.decoder.decoder.vit.parameters():
                p.requires_grad = False
            for p in self.decoder.decoder.head.parameters():
                p.requires_grad = True
            for p in self.decoder.decoder.head_norm.parameters():
                p.requires_grad = True
            print("[ControlAE] Freeze ViT backbone (train head only)")
        if noise_config != '__none__':
            print('Using noise')
            self.noise = instantiate_from_config(noise_config)
        # copy weights from first stage
        self.control.copy_encoder_weight(self.ae)
        # freeze first stage
        self.ae.eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False

        self.loss_layer = instantiate_from_config(loss_config)

        # early training phase
        # self.fixed_input = True
        self.fixed_x = None
        self.fixed_img = None
        self.fixed_input_recon = None
        self.fixed_control = None
        self.fixed_input = True
        # patch-wise alpha warmup (só é usado se o self.control tiver alpha_max/alpha)
        self.patch_alpha_warmup = 5000  # steps (ajusta)    
        # secret warmup
        self.secret_warmup = secret_warmup
        self.secret_baselen = 2
        self.secret_len = control_config.params.secret_len
        if self.secret_warmup:
            assert self.secret_len == 2**(int(np.log2(self.secret_len)))

        self.use_ema = use_ema
        if self.use_ema:
            print('Using EMA')
            self.control_ema = LitEma(self.control)
            self.decoder_ema = LitEma(self.decoder)
            print(f"Keeping EMAs of {len(list(self.control_ema.buffers()) + list(self.decoder_ema.buffers()))}.")

        if ckpt_path != '__none__':
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
    
    def on_train_batch_start(self, batch, batch_idx):
        if self.vit_freeze_steps <= 0:
            return
        if self._vit_unfrozen:
            return
        if self.global_step < self.vit_freeze_steps:
            return

        if not self._has_vit_backbone:
            self._vit_unfrozen = True
            return

        for p in self.decoder.decoder.vit.parameters():
            p.requires_grad = True

        self._vit_unfrozen = True
        print(f"[ControlAE] Unfreeze ViT backbone @ step {self.global_step}")

    def get_warmup_secret(self, old_secret):
        # old_secret: [B, secret_len]
        # new_secret: [B, secret_len]
        if self.secret_warmup:
            bsz = old_secret.shape[0]
            nrepeats = self.secret_len // self.secret_baselen
            new_secret  = torch.zeros((bsz, self.secret_baselen), dtype=torch.float).random_(0, 2).repeat_interleave(nrepeats, dim=1)
            return new_secret.to(old_secret.device)
        else:
            return old_secret
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.control_ema.store(self.control.parameters())
            self.decoder_ema.store(self.decoder.parameters())
            self.control_ema.copy_to(self.control)
            self.decoder_ema.copy_to(self.decoder)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.control_ema.restore(self.control.parameters())
                self.decoder_ema.restore(self.decoder.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.control_ema(self.control)
            self.decoder_ema(self.decoder)

    def compute_loss(self, pred, target):
        # return thf.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
        lpips_loss = self.lpips_loss(pred, target).mean(dim=[1,2,3])
        pred_yuv = color.rgb_to_yuv((pred + 1) / 2)
        target_yuv = color.rgb_to_yuv((target + 1) / 2)
        yuv_loss = torch.mean((pred_yuv - target_yuv)**2, dim=[2,3])
        yuv_loss = 1.5*torch.mm(yuv_loss, self.yuv_scales).squeeze(1)
        return lpips_loss + yuv_loss

    def forward(self, x, image, c):
        cname = self.control.__class__.__name__
        if cname in ['SecretEncoder6', 'SecretEncoderPatch']:
            eps, posterior = self.control(x, c)       # LATENT
        else:
            eps, posterior = self.control(image, c)   # RGB (comportamento antigo)
        return x + eps, posterior

    @torch.no_grad()
    def get_input(self, batch, return_first_stage=False, bs=None):
        image = batch[self.first_stage_key]
        control = batch[self.control_key]
        control = self.get_warmup_secret(control)
        if bs is not None:
            image = image[:bs]
            control = control[:bs]
        else:
            bs = image.shape[0]
        # encode image 1st stage
        image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
        x = self.encode_first_stage(image).detach()
        image_rec = self.decode_first_stage(x).detach()
        
        # check if using fixed input (early training phase)
        # if self.training and self.fixed_input:
        if self.fixed_input:
            if self.fixed_x is None:  # first iteration
                print('[TRAINING] Warmup - using fixed input image for now!')
                self.fixed_x = x.detach().clone()[:bs]
                self.fixed_img = image.detach().clone()[:bs]
                self.fixed_input_recon = image_rec.detach().clone()[:bs]
                self.fixed_control = control.detach().clone()[:bs]  # use for log_images with fixed_input option only
            x, image, image_rec = self.fixed_x, self.fixed_img, self.fixed_input_recon
        
        out = [x, control]
        if return_first_stage:
            out.extend([image, image_rec])
        return out

    def decode_first_stage(self, z):
        z = 1./self.scale_factor * z
        image_rec = self.ae.decode(z)
        return image_rec
    
    def encode_first_stage(self, image):
        encoder_posterior = self.ae.encode(image)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    def _update_patch_alpha(self):
        # Só aplica se o encoder tiver alpha_max (ex: SecretEncoderPatch com rampa)
        if not hasattr(self.control, "alpha_max"):
            return

        warm = float(getattr(self, "patch_alpha_warmup", 0))
        if warm <= 0:
            t = 1.0
        else:
            t = min(1.0, float(self.global_step) / warm)

        self.control.alpha = self.control.alpha_max * t
        if not hasattr(self, "_printed_alpha"):
            print(f"[PatchAlpha] warmup={warm} alpha_max={self.control.alpha_max} alpha={self.control.alpha:.4f}")
            self._printed_alpha = True
            
    def shared_step(self, batch):
        self._update_patch_alpha()   # <-- METE ISTO AQUI (primeira linha)

        x, c, img, _ = self.get_input(batch, return_first_stage=True)
        # import pdb; pdb.set_trace()
        x, posterior = self(x, img, c)
        image_rec = self.decode_first_stage(x)

        # clamp logo após reconstrução
        image_rec = image_rec.clamp(-1.0, 1.0)

        # resize
        if img.shape[-1] > 256:
            img = thf.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False).detach()
            image_rec = thf.interpolate(image_rec, size=(256, 256), mode='bilinear', align_corners=False)

        # noise / ataques
        if hasattr(self, 'noise') and self.noise.is_activated():
            image_rec_noised = self.noise(image_rec, self.global_step, p=0.9)
        else:
            image_rec_noised = image_rec

        # clamp após ataques (muito importante se JPEG/noise empurra fora)
        image_rec_noised = image_rec_noised.clamp(-1.0, 1.0)

        if self.global_step == 0:
            print("image_rec range (clamped):", image_rec.min().item(), image_rec.max().item())
            print("image_rec_noised range (clamped):", image_rec_noised.min().item(), image_rec_noised.max().item())

        pred = self.decoder(image_rec_noised)

        loss, loss_dict = self.loss_layer(img, image_rec, posterior, c, pred, self.global_step)
        bit_acc = loss_dict["bit_acc"]

        bit_acc_ = bit_acc.item()

        if (bit_acc_ > 0.98) and (not self.fixed_input) and self.noise.is_activated():
            self.loss_layer.activate_ramp(self.global_step)

        if (bit_acc_ > 0.95) and (not self.fixed_input):  # ramp up image loss at late training stage
            if hasattr(self, 'noise') and (not self.noise.is_activated()):
                self.noise.activate(self.global_step) 
        
        # if (bit_acc_ > 0.95) and (not self.fixed_input) and self.secret_warmup:
        #     if self.secret_baselen == self.secret_len:  # warm up done
        #         self.secret_warmup = False
        #     else:
        #         print(f'[TRAINING] secret length warmup: {self.secret_baselen} -> {self.secret_baselen*2}')
        #         self.secret_baselen *= 2

        if (bit_acc_ > 0.9) and self.fixed_input:  # execute only once
            print(f'[TRAINING] High bit acc ({bit_acc_}) achieved, switch to full image dataset training.')
            self.fixed_input = False
        return loss, loss_dict

    def _prefix_keys(self, d: dict, prefix: str, drop_keys=("img_lw",)):
        out = {}
        for k, v in d.items():
            if k in drop_keys:
                continue
            # se já vier com train/ ou val/, remove antes de colocar o prefix correto
            if k.startswith("train/"):
                k = k[len("train/"):]
            elif k.startswith("val/"):
                k = k[len("val/"):]
            out[f"{prefix}/{k}"] = v
        return out


    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        loss_dict = self._prefix_keys(loss_dict, "train")
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("train/global_step", float(self.global_step), prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        loss_dict_no_ema = self._prefix_keys(loss_dict_no_ema, "val")

        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = self._prefix_keys(loss_dict_ema, "val")
            # marca como EMA
            loss_dict_ema = {k + "_ema": v for k, v in loss_dict_ema.items()}

        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    @torch.no_grad()
    def log_images(self, batch, fixed_input=False, **kwargs):
        log = dict()
        if fixed_input and self.fixed_img is not None:
            x, c, img, img_recon = self.fixed_x, self.fixed_control, self.fixed_img, self.fixed_input_recon
        else:
            x, c, img, img_recon = self.get_input(batch, return_first_stage=True)
        x, _ = self(x, img, c)
        image_out = self.decode_first_stage(x)
        if hasattr(self, 'noise') and self.noise.is_activated():
            img_noise = self.noise(image_out, self.global_step, p=1.0)
            log['noised'] = img_noise
        log['input'] = img
        log['output'] = image_out
        log['recon'] = img_recon
        return log
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer
    



