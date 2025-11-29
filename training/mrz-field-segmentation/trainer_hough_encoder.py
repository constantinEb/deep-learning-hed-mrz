import csv
import math
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_hough_encoder as config

class CUDAPrefetcher:
    def __init__(self, loader, device, transform=None):
        self.loader_iter = iter(loader)
        self.device = device
        self.transform = transform
        self.stream = torch.cuda.Stream() if "cuda" in device else None
        self.next_inputs = None
        self.next_targets = None
        self._preload()

    def _to_device(self, x):
        return x.pin_memory().to(self.device, non_blocking=True) if hasattr(x, "pin_memory") else x.to(self.device, non_blocking=True)

    def _preload(self):
        try:
            imgs, msks = next(self.loader_iter)
        except StopIteration:
            self.next_inputs = None
            self.next_targets = None
            return
        if self.stream is None:
            self.next_inputs, self.next_targets = imgs, msks
            return
        with torch.cuda.stream(self.stream):
            imgs = self._to_device(imgs)
            msks = self._to_device(msks)
            if self.transform is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    imgs, msks = self.transform(imgs, msks)
            if imgs.dtype != torch.float32:
                imgs = imgs.float()
            self.next_inputs, self.next_targets = imgs, msks

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_inputs is None:
            raise StopIteration
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        imgs, msks = self.next_inputs, self.next_targets
        self._preload()
        return imgs, msks

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        t = torch.from_numpy(arr)[None, ...]  # (1,H,W)
    else:
        t = torch.from_numpy(arr).permute(2, 0, 1)
    return t

def iou_binary(pred: torch.Tensor, targ: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred * targ).sum().item()
    union = (pred + targ - pred * targ).sum().item()
    return inter / (union + eps)

def _build_grids(angles_deg: torch.Tensor, Hp: int, Wp: int, device, sign: int) -> torch.Tensor:
    thetas = torch.deg2rad(angles_deg.to(device))
    grids = []
    for th in thetas:
        c, s = torch.cos(sign * th), torch.sin(sign * th)
        rot = torch.tensor([[c, -s, 0.0],
                            [s,  c, 0.0]], dtype=torch.float32, device=device)
        g = F.affine_grid(rot[None], size=(1, 1, Hp, Wp), align_corners=False)
        grids.append(g[0])
    return torch.stack(grids, dim=0)  # (A,Hp,Wp,2)


class RadonLayer(nn.Module):
    """
    Differentiable Radon-like transform:
      - Rotate feature map for each angle
      - Sum along x-axis → projection ρ
    Output: (B, Hr, A), where Hr is padded height.
    """
    def __init__(self, n_angles: int = 64, pad_ratio: float = 1.0):
        super().__init__()
        self.n_angles = n_angles
        self.register_buffer("angles_deg", torch.linspace(0., 180., steps=n_angles + 1)[:-1])
        self.pad_ratio = pad_ratio
        self._grid_cache: Dict[Tuple[int, int, str, int], torch.Tensor] = {}

    def _get_grids(self, Hp: int, Wp: int, device, sign: int) -> torch.Tensor:
        key = (Hp, Wp, str(device), sign)
        if key not in self._grid_cache:
            self._grid_cache[key] = _build_grids(self.angles_deg, Hp, Wp, device, sign)
        g = self._grid_cache[key]
        if g.device != device:
            g = g.to(device)
            self._grid_cache[key] = g
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W)
        B, _, H, W = x.shape
        x = x.float()
        pad = int(self.pad_ratio * 0.1 * min(H, W))
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        _, _, Hp, Wp = x.shape

        grids = self._get_grids(Hp, Wp, x.device, sign=+1)          # (A,Hp,Wp,2)
        x_rep = x.repeat(self.n_angles, 1, 1, 1)                    # (A*B,1,Hp,Wp)
        G = grids.unsqueeze(1).repeat(1, B, 1, 1, 1).view(-1, Hp, Wp, 2)
        xr = F.grid_sample(x_rep, G, mode="bilinear",
                           padding_mode="zeros", align_corners=False)  # (A*B,1,Hp,Wp)
        proj = xr.sum(dim=3) / float(Wp)                             # (A*B,1,Hp)
        sino = proj.view(self.n_angles, B, 1, Hp).permute(1, 3, 0, 2)  # (B,Hp,A,1)
        return sino.squeeze(3)                                       # (B,Hp,A)


class BackProjectionLayer(nn.Module):
    """
    Transposed Radon/Hough:
      - Expand projections to (Hp,Wp)
      - Rotate back by -angle
      - Sum over angles; crop padding; normalize by #angles
    """
    def __init__(self, angles_deg: torch.Tensor, pad_ratio: float = 1.0):
        super().__init__()
        self.register_buffer("angles_deg", angles_deg.clone())
        self.pad_ratio = pad_ratio
        self._grid_cache: Dict[Tuple[int, int, str, int], torch.Tensor] = {}

    def _get_grids(self, Hp: int, Wp: int, device, sign: int) -> torch.Tensor:
        key = (Hp, Wp, str(device), sign)
        if key not in self._grid_cache:
            self._grid_cache[key] = _build_grids(self.angles_deg, Hp, Wp, device, sign)
        g = self._grid_cache[key]
        if g.device != device:
            g = g.to(device)
            self._grid_cache[key] = g
        return g

    def forward(self, sino: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        # sino: (B, A, Hr)
        B, A, Hr = sino.shape
        H, W = out_hw
        pad = int(self.pad_ratio * 0.1 * min(H, W))
        Hp, Wp = H + 2 * pad, W + 2 * pad

        proj = sino.float().permute(1, 0, 2).unsqueeze(3)             # (A,B,Hr,1)
        proj = F.interpolate(proj, size=(Hp, Wp), mode="bilinear", align_corners=False)
        proj = proj.reshape(A * B, 1, Hp, Wp)

        grids = self._get_grids(Hp, Wp, proj.device, sign=-1)         # (A,Hp,Wp,2)
        G = grids.unsqueeze(1).repeat(1, B, 1, 1, 1).view(-1, Hp, Wp, 2)
        xr = F.grid_sample(proj, G, mode="bilinear",
                           padding_mode="zeros", align_corners=False)  # (A*B,1,Hp,Wp)
        acc = xr.view(A, B, 1, Hp, Wp).sum(dim=0) / float(A + 1e-6)   # (B,1,Hp,Wp)
        if pad > 0:
            acc = acc[:, :, pad:-pad, pad:-pad]
        return acc                                                    # (B,1,H,W)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.pad = nn.ReflectionPad2d(k // 2)
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(self.pad(x))))


class Conv3x3RP(nn.Module):
    """3×3 conv with reflection padding + BN + ReLU."""
    def __init__(self, cin, cout):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Conv1x1RP(nn.Module):
    """1×1 conv wrapped to keep API shape-consistent."""
    def __init__(self, cin, cout, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(0)
        self.conv = nn.Conv2d(cin, cout, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class HEDMRZ(nn.Module):
    """
    Heavier HED-MRZ with HoughEncoder-style backbone:

      Encoder (U-Net style, base_ch=32):
        enc1: 1   → 32 → 32 @ 384×384
        enc2: 32  → 64 → 64 @ 192×192
        enc3: 64  → 128 → 128 @ 96×96
        bottleneck: 128 → 256 → 256 @ 48×48

      Hough block on enc3:
        enc3 (B,128,H3,W3)
          → 1×1 reduce to 1ch → Softsign
          → resize to fht_size×fht_size (default 128×128)
          → Radon(FHT) → 3× Conv3×3 in Hough space (sino_ch channels)
          → 1×1 to 1ch → BackProjection
          → 1×1 to 128ch → upsample to H3×W3
          → fused with enc3 by SUM

      Decoder:
        up3: 256 → 128, concat with fused enc3 (128) → 256 → 128
        up2: 128 → 64,  concat with enc2 (64)       → 128 → 64
        up1: 64  → 32,  concat with enc1 (32)       → 64  → 32
        head: 32 → 1 logits
    """
    def __init__(self, img_size=384, n_angles=128, sino_ch=32, base_ch=32, fht_size=128):
        super().__init__()
        self.img_size = img_size
        self.fht_size = fht_size

        self.radon = RadonLayer(n_angles=n_angles, pad_ratio=config.radon_pad)
        self.backproj = BackProjectionLayer(self.radon.angles_deg, pad_ratio=config.radon_pad)

        self.enc1 = nn.Sequential(
            ConvBlock(1, base_ch),
            ConvBlock(base_ch, base_ch),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(base_ch, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 2),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 4),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(base_ch * 4, base_ch * 8),
            ConvBlock(base_ch * 8, base_ch * 8),
        )

        self.h_pre_reduce = Conv1x1RP(base_ch * 4, 1, bias=False)
        self.h_soft = nn.Softsign()
        self.h_in = Conv3x3RP(1, sino_ch)
        self.h_mid1 = Conv3x3RP(sino_ch, sino_ch)
        self.h_mid2 = Conv3x3RP(sino_ch, sino_ch)
        self.h_out = Conv1x1RP(sino_ch, 1, bias=False)
        self.h_post = Conv1x1RP(1, base_ch * 4, bias=False)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(base_ch * 8, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 4),
        )

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(base_ch * 4, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 2),
        )

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch),
            ConvBlock(base_ch, base_ch),
        )

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

        # Init conv weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)                   # (B, base, 384,384)
        x2 = self.enc2(self.pool1(x1))      # (B, 2*base, 192,192)
        x3 = self.enc3(self.pool2(x2))      # (B, 4*base, 96,96)
        x4 = self.bottleneck(self.pool3(x3))# (B, 8*base, 48,48)

        # Hough block on x3
        h = self.h_pre_reduce(x3)           # (B,1,H3,W3)
        h = self.h_soft(h)
        h = F.adaptive_avg_pool2d(h, (self.fht_size, self.fht_size))  # (B,1,fht,fht)

        sino = self.radon(h)                # (B,Hr,A)
        sino_4d = sino.unsqueeze(1)         # (B,1,Hr,A)

        s = self.h_in(sino_4d)
        s = self.h_mid1(s)
        s = self.h_mid2(s)
        s = self.h_out(s)                   # (B,1,Hr,A)

        sino_ba = s.squeeze(1).permute(0, 2, 1)       # (B,A,Hr)
        hmap = self.backproj(sino_ba, (self.fht_size, self.fht_size))  # (B,1,fht,fht)
        hmap = self.h_post(hmap)                       # (B,4*base,fht,fht)
        hmap = F.interpolate(hmap, size=x3.shape[2:], mode="bilinear", align_corners=False)

        x3_fused = x3 + hmap

        # Decoder
        u3 = self.up3(x4)                  # (B,4*base,96,96)
        u3 = torch.cat([u3, x3_fused], dim=1)  # (B,8*base,96,96)
        d3 = self.dec3(u3)                 # (B,4*base,96,96)

        u2 = self.up2(d3)                  # (B,2*base,192,192)
        u2 = torch.cat([u2, x2], dim=1)    # (B,4*base,192,192)
        d2 = self.dec2(u2)                 # (B,2*base,192,192)

        u1 = self.up1(d2)                  # (B,base,384,384)
        u1 = torch.cat([u1, x1], dim=1)    # (B,2*base,384,384)
        d1 = self.dec1(u1)                 # (B,base,384,384)

        logits = self.head(d1)             # (B,1,384,384)
        return logits

class WeightedBCE(nn.Module):
    def __init__(self, bg_weight=0.5, pos_weight=1.0):
        super().__init__()
        self.bg_weight = float(bg_weight)
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, targets):
        # logits/targets: (B,1,H,W)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        w_bg = torch.full_like(bce, self.bg_weight)
        w = torch.where(targets > 0.5, torch.ones_like(bce), w_bg)
        return (bce * w).mean()


class BCEDice(nn.Module):
    def __init__(self, bg_weight=0.5, pos_weight=2.0, eps=1e-6):
        super().__init__()
        self.bce = WeightedBCE(bg_weight, pos_weight)
        self.eps = eps

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice_loss = 1 - (2 * inter + self.eps) / (union + self.eps)
        return 0.5 * bce_loss + 0.5 * dice_loss

def postprocess_to_obbs(prob_map_np, orig_hw):
    H0, W0 = orig_hw
    hm = cv2.resize(prob_map_np, (W0, H0), interpolation=cv2.INTER_LINEAR)
    binm = (hm >= 0.5).astype('uint8') * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    obbs = []
    min_area = 0.01 * H0 * W0
    for i in range(1, n):
        comp = (labels == i).astype('uint8') * 255
        if comp.sum() < min_area:
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        (w, h) = rect[1]
        ar = max(w, h) / max(1e-6, min(w, h))
        if ar < 7.0:
            continue
        obb = cv2.boxPoints(rect).astype('float32')
        obbs.append(obb)
    return obbs

@torch.no_grad()
def validate_seg(model, loader, device, thr=0.5):
    model.eval()
    ious = []
    for img, msk in loader:
        img = img.to(device, non_blocking=True)
        msk = msk.to(device, non_blocking=True)
        if img.shape[1] != 1:
            img = img.mean(dim=1, keepdim=True)
        logits = model(img)
        preds = (torch.sigmoid(logits) > thr).float()
        inter = (preds * msk).sum(dim=(1, 2, 3))
        union = (preds + msk - preds * msk).sum(dim=(1, 2, 3))
        ious.extend(((inter + 1e-6) / (union + 1e-6)).cpu().tolist())
    return float(torch.tensor(ious).mean())

@torch.no_grad()
def validate(model, loader, device, thr=0.5, use_postprocess=False):
    model.eval()
    ious = []
    for img, msk in loader:
        img = img.to(device, non_blocking=True)
        msk = msk.to(device, non_blocking=True)
        if img.shape[1] != 1:
            img = img.mean(dim=1, keepdim=True)
        logits = model(img)

        if use_postprocess:
            B, _, H, W = logits.shape
            for b in range(B):
                prob_map = torch.sigmoid(logits[b, 0]).cpu().numpy()
                obbs = postprocess_to_obbs(prob_map, orig_hw=(H, W))

                pred_mask = np.zeros((H, W), dtype=np.uint8)
                for obb in obbs:
                    cv2.fillPoly(pred_mask, [obb.astype(np.int32)], 255)
                pred_mask = (pred_mask > 0).astype(np.float32)

                pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).to(device)
                ious.append(iou_binary(pred_tensor, msk[b:b+1]))
        else:
            preds = (torch.sigmoid(logits) > thr).float()
            ious.append(iou_binary(preds, msk))

    return float(np.mean(ious)) if ious else 0.0

def train():
    from dataset import create_dataloaders
    from transformer import get_train_transform, get_val_transform

    set_seed(config.split_seed)

    config.out_dir = config.generate_run_dir()
    ensure_dir(config.out_dir)
    print(f"Saving run to: {config.out_dir}")

    # Copy config file to run directory for reproducibility
    config_src = os.path.join(os.path.dirname(__file__), "config_hough_encoder.py")
    config_dst = os.path.join(config.out_dir, "config_hough_encoder.py")
    shutil.copy(config_src, config_dst)
    print(f"Config copied to: {config_dst}")

    torch.backends.cudnn.benchmark = True

    aug_config = config.get_aug_config()

    train_loader, val_loader, test_loader = create_dataloaders(
        datasets=["MIDV500", "MIDV2020"],
        data_roots={"MIDV500": "../../data/midv500", "MIDV2020": "../../data/midv2020"},
        split_method="random",
        batch_size=config.batch_size,
        img_size=config.img_size,
        grayscale=config.grayscale,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        use_kornia=True,
        midv2020_modalities=config.modalities,
    )

    train_transform = get_train_transform(
        img_size=config.img_size,
        aug_config=aug_config,
        device=config.device
    )
    val_transform = get_val_transform(
        img_size=config.img_size,
        device=config.device
    )

    model = HEDMRZ(
        img_size=config.img_size,
        n_angles=config.n_angles,
        sino_ch=config.h_sino_channels,
        base_ch=32,
        fht_size=128,
    ).to(config.device)

    criterion = WeightedBCE(bg_weight=config.bg_weight, pos_weight=config.pos_weight).to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=config.lr,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    best_iou = 0.0

    log_path = os.path.join(config.out_dir, config.log_file)
    csv_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(['epoch', 'train_loss', 'val_mIoU_seg',
                             'val_mIoU_obb', 'best_iou_seg', 'epoch_time_s'])

    print(f"Logging training results to: {log_path}")

    if config.profile_timing:
        print("=" * 70)
        print("Epoch 1 timing breakdown will be reported")
        print("=" * 70)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0

        times = {"data": 0.0, "augment": 0.0, "forward": 0.0, "backward": 0.0, "validation": 0.0}
        t_epoch_start = time.time()

        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch:04d}/{config.epochs}",
                    unit="batch", leave=False)

        prefetch = CUDAPrefetcher(train_loader, config.device, transform=train_transform)

        for img, msk in prefetch:
            if img.shape[1] != 1:
                img = img.mean(dim=1, keepdim=True)

            optim.zero_grad(set_to_none=True)

            t0 = time.time()
            logits = model(img)
            loss = criterion(logits, msk)
            if config.device == "cuda":
                torch.cuda.synchronize()
            times["forward"] += time.time() - t0

            t1 = time.time()
            loss.backward()
            optim.step()
            scheduler.step()
            if config.device == "cuda":
                torch.cuda.synchronize()
            times["backward"] += time.time() - t1

            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            pbar.update(1)


        pbar.close()

        # Validation
        t_val_start = time.time()
        miou_seg = validate_seg(model, val_loader, config.device, thr=config.threshold)
        miou_obb = validate(model, val_loader, config.device, thr=config.threshold, use_postprocess=True)
        times["validation"] = time.time() - t_val_start


        if config.profile_timing and epoch == 1:
            t_total = time.time() - t_epoch_start
            print("\n" + "=" * 70)
            print("TIMING BREAKDOWN (Epoch 1)")
            for k in ("data", "augment", "forward", "backward", "validation"):
                pct = 100.0 * times[k] / max(1e-9, t_total)
                print(f"  {k:<12}: {times[k]:.2f}s ({pct:.1f}%)")
            print(f"  Total epoch: {t_total:.2f}s")
            print("=" * 70 + "\n")

        avg_loss = running / max(1, len(train_loader))
        epoch_time = time.time() - t_epoch_start
        print(f"Epoch {epoch:04d} | train_loss {avg_loss:.4f} | "
              f"val_mIoU_seg {miou_seg:.4f} | val_mIoU_obb {miou_obb:.4f}")

        if miou_seg > best_iou:
            best_iou = miou_seg

        with open(log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch,
                f"{avg_loss:.6f}",
                f"{miou_seg:.6f}",
                f"{miou_obb:.6f}",
                f"{best_iou:.6f}",
                f"{epoch_time:.2f}",
            ])

        if miou_seg == best_iou:
            torch.save({
                "model": model.state_dict(),
                "cfg": config.to_dict(),
                "val_mIoU_seg": miou_seg,
                "val_mIoU_obb": miou_obb,
            }, os.path.join(config.out_dir, "best.pt"))

        if epoch % config.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "cfg": config.to_dict(),
                "epoch": epoch,
            }, os.path.join(config.out_dir, f"epoch_{epoch}.pt"))

    torch.save({
        "model": model.state_dict(),
        "cfg": config.to_dict(),
    }, os.path.join(config.out_dir, "last.pt"))

    print("\n" + "=" * 70)
    print("Training completed! Running inference on test set...")
    print("=" * 70)

    # Run inference on test set with best model
    best_model_path = os.path.join(config.out_dir, "best.pt")
    inference_pdf_path = os.path.join(config.out_dir, "inference_results_test.pdf")

    if os.path.exists(best_model_path):
        # Load best model
        best_checkpoint = torch.load(best_model_path, map_location=config.device)
        model.load_state_dict(best_checkpoint["model"])
        model.eval()

        # Import inference function
        from inference_test import generate_pdf_report

        # Determine number of test samples (up to 100)
        num_test_samples = min(100, len(test_loader.dataset))

        # Generate checkpoint info
        checkpoint_info = f"Best model checkpoint | val_mIoU_seg: {best_checkpoint.get('val_mIoU_seg', 'N/A'):.4f}"

        # Generate PDF report
        generate_pdf_report(
            model=model,
            test_loader=test_loader,
            output_path=inference_pdf_path,
            num_samples=num_test_samples,
            device=config.device,
            checkpoint_info=checkpoint_info
        )

        print(f"\nInference PDF saved to: {inference_pdf_path}")
    else:
        print(f"Warning: Best model not found at {best_model_path}, skipping inference.")


@torch.no_grad()
def infer_logits(model: nn.Module, img: Image.Image) -> torch.Tensor:
    im = ImageOps.grayscale(img) if config.grayscale else img.convert("RGB")
    im = im.resize((config.img_size, config.img_size), Image.BILINEAR)
    t = to_tensor(im).unsqueeze(0).float().to(config.device) / 255.0
    if t.shape[1] != 1:
        t = t.mean(dim=1, keepdim=True)
    logits = model(t)
    return logits  # (1,1,H,W)


@torch.no_grad()
def infer_prob_map(model: nn.Module, img: Image.Image) -> np.ndarray:
    logits = infer_logits(model, img)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return prob


@torch.no_grad()
def infer_with_postprocess(model: nn.Module, img: Image.Image) -> tuple:
    orig_size = img.size  # (W, H)
    prob_map_384 = infer_prob_map(model, img)
    prob_map_full = cv2.resize(prob_map_384, orig_size, interpolation=cv2.INTER_LINEAR)
    obbs = postprocess_to_obbs(prob_map_full, orig_hw=(orig_size[1], orig_size[0]))
    return obbs, prob_map_full

if __name__ == "__main__":
    train()
