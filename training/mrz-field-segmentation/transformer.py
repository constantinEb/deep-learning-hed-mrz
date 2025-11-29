import torch
import torch.nn as nn
import kornia.augmentation as K
from torch import amp
from typing import Dict, Tuple, Optional
import torchvision.transforms.functional as TF
from io import BytesIO
from PIL import Image
import numpy as np


def apply_jpeg_compression_gpu(img: torch.Tensor, quality_range: Tuple[int, int] = (60, 100)) -> torch.Tensor:
    device = img.device
    B, C, H, W = img.shape

    img_cpu = img.cpu()

    compressed_imgs = []
    for i in range(B):
        quality = torch.randint(quality_range[0], quality_range[1] + 1, (1,)).item()

        img_np = img_cpu[i].numpy()

        if C == 1:
            img_np = np.repeat(img_np, 3, axis=0)

        img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, mode='RGB')

        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_pil_compressed = Image.open(buffer)

        img_np_compressed = np.array(img_pil_compressed).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np_compressed).permute(2, 0, 1)

        # Convert back to grayscale if needed
        if C == 1:
            img_tensor = img_tensor.mean(dim=0, keepdim=True)

        compressed_imgs.append(img_tensor)

    result = torch.stack(compressed_imgs, dim=0).to(device)
    return result


class MRZTransform(nn.Module):
    def __init__(
        self,
        img_size: int = 384,
        train: bool = True,
        aug_config: Optional[Dict] = None
    ):
        super().__init__()
        self.img_size = img_size
        self.train = train
        self.aug_config = aug_config or {}

        if train:
            self.geo_transforms, self.photo_transforms, self.aug_prob = self._create_train_transform()
            self.resize_only = self._create_val_transform()
        else:
            self.geo_transforms = None
            self.photo_transforms = None
            self.jpeg_quality = None
            self.aug_prob = 0.0
            self.resize_only = self._create_val_transform()

    def _create_train_transform(self) -> Tuple[nn.Sequential, nn.Sequential, float]:
        # - 95% of batches get full augmentation (geometric + photometric)
        # - 5% of batches get only resize (no augmentation)
        aug_prob = self.aug_config.get("aug_prob", 0.95)

        pers_range = self.aug_config.get("aug_perspective_scale", (0.05, 0.12))
        self.pers_lo = float(pers_range[0])
        self.pers_hi = float(pers_range[1])

        geo = nn.Sequential(
            K.Resize(
                size=(self.img_size, self.img_size),
                align_corners=False,
                keepdim=True
            ),
            K.RandomPerspective(
                distortion_scale=0.05,
                p=1.0,
                keepdim=True,
                same_on_batch=False
            ),
            K.RandomRotation(
                degrees=float(self.aug_config.get('aug_rotate_limit', 25)),
                p=1.0,
                keepdim=True,
                same_on_batch=False
            )
        )

        photo_augs = [
            K.RandomMotionBlur(
                kernel_size=self.aug_config.get('aug_blur_limit', 7),
                angle=(-45, 45),
                direction=(-1.0, 1.0),
                p=1.0,
                keepdim=True,
                same_on_batch=False
            ),
            K.RandomBrightness(
                brightness=(0.7, 1.3),
                p=1.0,
                keepdim=True,
                same_on_batch=False
            ),
            K.RandomContrast(
                contrast=(0.7, 1.3),
                p=1.0,
                keepdim=True,
                same_on_batch=False
            )
        ]

        if hasattr(K, 'RandomGaussianNoise'):
            photo_augs.append(
                K.RandomGaussianNoise(
                    mean=0.0,
                    std=0.01,
                    p=1.0,
                    keepdim=True,
                    same_on_batch=False
                )
            )

        self.jpeg_quality = self.aug_config.get('aug_compression_quality', (60, 100))

        photo = nn.Sequential(*photo_augs)

        return geo, photo, aug_prob

    def _create_val_transform(self) -> nn.Sequential:
        return nn.Sequential(
            K.Resize(
                size=(self.img_size, self.img_size),
                align_corners=False,
                keepdim=True
            )
        )

    def forward(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.float()

        apply_aug = self.train and (torch.rand(1).item() < self.aug_prob)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            mask = mask.float()

            if apply_aug:
                combined = torch.cat([image, mask], dim=1)


                perspective = self.geo_transforms[1]

                distortion_scale = float(
                    self.pers_lo + torch.rand(1, dtype=torch.float32, device=image.device).item() * (self.pers_hi - self.pers_lo)
                )
                perspective.distortion_scale = distortion_scale

                from torch.distributions.uniform import Uniform
                low = torch.tensor(0.0, device=image.device, dtype=torch.float32)
                high = torch.tensor(1.0, device=image.device, dtype=torch.float32)
                perspective.rand_val_sampler = Uniform(low, high)

                combined_aug = self.geo_transforms(combined)

                img_aug = combined_aug[:, :-1, :, :]
                mask_aug = combined_aug[:, -1:, :, :]

                # Use custom CPU-based JPEG function (workaround for Kornia's device bug)
                # if self.jpeg_quality is not None:
                #    img_aug = apply_jpeg_compression_gpu(img_aug, self.jpeg_quality)

                img_aug = self.photo_transforms(img_aug)

                # Binarize mask after geometric transforms (values may have been interpolated)
                mask_aug = (mask_aug > 0.5).float()
            else:
                # No augmentation: just resize
                combined = torch.cat([image, mask], dim=1)
                combined_resized = self.resize_only(combined)
                img_aug = combined_resized[:, :-1, :, :]
                mask_aug = combined_resized[:, -1:, :, :]
                mask_aug = (mask_aug > 0.5).float()

            return img_aug, mask_aug

        else:
            if apply_aug:
                perspective = self.geo_transforms[1]

                distortion_scale = float(
                    self.pers_lo + torch.rand(1, dtype=torch.float32, device=image.device).item() * (self.pers_hi - self.pers_lo)
                )
                perspective.distortion_scale = distortion_scale

                from torch.distributions.uniform import Uniform
                low = torch.tensor(0.0, device=image.device, dtype=torch.float32)
                high = torch.tensor(1.0, device=image.device, dtype=torch.float32)
                perspective.rand_val_sampler = Uniform(low, high)

                img_aug = self.geo_transforms(image)

                if hasattr(K, 'RandomJPEG'):
                    x3 = img_aug.repeat(1, 3, 1, 1)
                    jpeg_quality = self.aug_config.get('aug_compression_quality', (60, 100))
                    x3 = K.RandomJPEG(jpeg_quality=jpeg_quality, p=1.0, keepdim=True, same_on_batch=False)(x3)
                    img_aug = x3.mean(dim=1, keepdim=True)

                img_aug = self.photo_transforms(img_aug)
            else:
                img_aug = self.resize_only(image)

            return img_aug, None

    def __repr__(self) -> str:
        mode = "train" if self.train else "val"
        if self.train:
            n_geo = len(self.geo_transforms) if self.geo_transforms else 0
            n_photo = len(self.photo_transforms) if self.photo_transforms else 0
            return f"MRZTransform(mode={mode}, img_size={self.img_size}, aug_prob={self.aug_prob:.2f}, geo={n_geo}, photo={n_photo})"
        else:
            return f"MRZTransform(mode={mode}, img_size={self.img_size})"


def get_train_transform(
    img_size: int = 384,
    aug_config: Optional[Dict] = None,
    device: str = "cuda"
) -> MRZTransform:
    transform = MRZTransform(img_size=img_size, train=True, aug_config=aug_config)
    return transform.to(device)


def get_val_transform(
    img_size: int = 384,
    device: str = "cuda"
) -> MRZTransform:
    transform = MRZTransform(img_size=img_size, train=False, aug_config=None)
    return transform.to(device)
