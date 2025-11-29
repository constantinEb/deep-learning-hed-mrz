import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class BaseDataCollector(ABC):
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.samples = []

    @abstractmethod
    def collect_samples(self, filter_passport: bool = True) -> List[Dict]:
        pass

    @abstractmethod
    def is_passport(self, doc_type: str) -> bool:
        pass

    def get_document_types(self) -> List[str]:
        if not self.samples:
            self.collect_samples(filter_passport=True)

        doc_types = sorted(set(s['doc_type'] for s in self.samples))
        return doc_types


class MIDV2020DataCollector(BaseDataCollector):

    def __init__(
        self,
        data_root: str,
        modalities: Optional[List[str]] = None
    ):
        super().__init__(data_root)

        if modalities is None:
            modalities = ["photo", "scan_upright", "scan_rotated", "clips"]

        self.modalities = modalities

    def is_passport(self, doc_type: str) -> bool:
        return 'passport' in doc_type.lower()

    def collect_samples(self, filter_passport: bool = True) -> List[Dict]:
        samples = []

        for modality in self.modalities:
            modality_path = self.data_root / modality

            if not modality_path.exists():
                print(f"Warning: Modality not found: {modality_path}")
                continue

            if modality == "clips":
                samples.extend(self._collect_clips_samples(filter_passport))
            else:
                samples.extend(self._collect_standard_samples(modality, filter_passport))

        self.samples = samples
        print(f"[MIDV2020] Collected {len(samples)} samples from {len(self.modalities)} modalities")

        return samples

    def _collect_standard_samples(self, modality: str, filter_passport: bool) -> List[Dict]:
        samples = []

        images_base = self.data_root / modality / 'images'
        masks_base = self.data_root / modality / 'annotations-hed-mrz'

        if not images_base.exists() or not masks_base.exists():
            print(f"Warning: Missing directories for {modality}")
            return samples

        for doc_type_dir in sorted(images_base.iterdir()):
            if not doc_type_dir.is_dir():
                continue

            doc_type = doc_type_dir.name

            if filter_passport and not self.is_passport(doc_type):
                continue

            mask_dir = masks_base / doc_type

            if not mask_dir.exists():
                print(f"Warning: No masks found for {modality}/{doc_type}")
                continue

            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(doc_type_dir.glob(ext))

            for img_path in image_files:
                mask_path = mask_dir / f"{img_path.stem}.png"

                if mask_path.exists():
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': str(mask_path),
                        'doc_type': doc_type,
                        'modality': modality,
                        'dataset': 'MIDV2020',
                        'clip_number': None
                    })

        print(f"  [MIDV2020] {modality}: {len(samples)} samples")
        return samples

    def _collect_clips_samples(self, filter_passport: bool) -> List[Dict]:
        samples = []

        images_base = self.data_root / 'clips' / 'images'
        masks_base = self.data_root / 'clips' / 'annotations-hed-mrz'

        if not images_base.exists() or not masks_base.exists():
            print(f"Warning: Missing directories for clips")
            return samples

        for doc_type_dir in sorted(images_base.iterdir()):
            if not doc_type_dir.is_dir():
                continue

            doc_type = doc_type_dir.name

            if filter_passport and not self.is_passport(doc_type):
                continue

            for clip_dir in sorted(doc_type_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue

                clip_number = clip_dir.name
                mask_dir = masks_base / doc_type / clip_number

                if not mask_dir.exists():
                    print(f"Warning: No masks found for clips/{doc_type}/{clip_number}")
                    continue

                image_extensions = ['*.jpg', '*.jpeg', '*.png']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(clip_dir.glob(ext))

                for img_path in image_files:
                    mask_path = mask_dir / f"{img_path.stem}.png"

                    if mask_path.exists():
                        samples.append({
                            'image_path': str(img_path),
                            'mask_path': str(mask_path),
                            'doc_type': doc_type,
                            'modality': 'clips',
                            'dataset': 'MIDV2020',
                            'clip_number': clip_number
                        })

        print(f"  [MIDV2020] clips: {len(samples)} samples")
        return samples


class MIDV500DataCollector(BaseDataCollector):
    CAPTURE_CONDITIONS = ['CA', 'CS', 'HA', 'HS', 'KA', 'KS', 'PA', 'PS', 'TA', 'TS']

    def __init__(
        self,
        data_root: str,
        conditions: Optional[List[str]] = None
    ):
        super().__init__(data_root)

        if conditions is None:
            conditions = self.CAPTURE_CONDITIONS

        self.conditions = conditions

    def is_passport(self, doc_type: str) -> bool:
        return 'passport' in doc_type.lower()

    def collect_samples(self, filter_passport: bool = True) -> List[Dict]:
        samples = []

        for doc_type_dir in sorted(self.data_root.iterdir()):
            if not doc_type_dir.is_dir():
                continue

            doc_type = doc_type_dir.name

            if filter_passport and not self.is_passport(doc_type):
                continue

            mask_base_dir = doc_type_dir / 'annotations-hed-mrz'
            if not mask_base_dir.exists():
                continue

            images_base = doc_type_dir / 'images'
            if not images_base.exists():
                continue

            for condition in self.conditions:
                condition_images_dir = images_base / condition

                if not condition_images_dir.exists():
                    continue

                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(condition_images_dir.glob(ext))

                for img_path in image_files:
                    mask_path = mask_base_dir / f"{img_path.stem}.png"

                    if mask_path.exists():
                        samples.append({
                            'image_path': str(img_path),
                            'mask_path': str(mask_path),
                            'doc_type': doc_type,
                            'condition': condition,
                            'dataset': 'MIDV500',
                        })

        self.samples = samples
        print(f"[MIDV500] Collected {len(samples)} samples from {len(self.conditions)} conditions")

        condition_counts = {}
        for s in samples:
            cond = s.get('condition', 'unknown')
            condition_counts[cond] = condition_counts.get(cond, 0) + 1

        for cond in sorted(condition_counts.keys()):
            print(f"  [MIDV500] {cond}: {condition_counts[cond]} samples")

        return samples


class MultiDatasetCollector:
    def __init__(self, collectors: List[BaseDataCollector]):
        self.collectors = collectors
        self.samples = []

    def collect_all_samples(self, filter_passport: bool = True) -> List[Dict]:
        all_samples = []

        for collector in self.collectors:
            samples = collector.collect_samples(filter_passport=filter_passport)
            all_samples.extend(samples)

        self.samples = all_samples

        dataset_counts = {}
        for s in all_samples:
            ds = s.get('dataset', 'unknown')
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

        for ds in sorted(dataset_counts.keys()):
            print(f"  {ds}: {dataset_counts[ds]} samples")

        return all_samples

    def get_document_types(self) -> List[str]:
        if not self.samples:
            self.collect_all_samples(filter_passport=True)

        doc_types = sorted(set(s['doc_type'] for s in self.samples))
        return doc_types

    def split_by_document_type(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[Dict]]:
        if not self.samples:
            self.collect_all_samples(filter_passport=True)

        doc_types = self.get_document_types()

        random.seed(seed)
        doc_types_shuffled = doc_types.copy()
        random.shuffle(doc_types_shuffled)

        n_types = len(doc_types)
        n_train = max(1, int(n_types * train_ratio))
        n_val = max(1, int(n_types * val_ratio))

        train_types = set(doc_types_shuffled[:n_train])
        val_types = set(doc_types_shuffled[n_train:n_train + n_val])
        test_types = set(doc_types_shuffled[n_train + n_val:])

        train_samples = [s for s in self.samples if s['doc_type'] in train_types]
        val_samples = [s for s in self.samples if s['doc_type'] in val_types]
        test_samples = [s for s in self.samples if s['doc_type'] in test_types]

        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        print(f"\nSample counts:")
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val: {len(val_samples)} samples")
        print(f"  Test: {len(test_samples)} samples")

        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }

    def split_random_by_clips(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        clip_subsample_rate: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Random split that keeps clip images together and subsamples clips.

        For MIDV2020 clips modality: groups all images from the same clip together and
        only includes every Nth image. All images from one clip stay in the same split.
        """
        if not self.samples:
            self.collect_all_samples(filter_passport=True)

        random.seed(seed)

        clip_groups = {}
        non_clip_samples = []

        for sample in self.samples:
            if sample.get('dataset') == 'MIDV2020' and sample.get('modality') == 'clips' and sample.get('clip_number') is not None:
                key = (sample['doc_type'], sample['clip_number'])
                if key not in clip_groups:
                    clip_groups[key] = []
                clip_groups[key].append(sample)
            else:
                non_clip_samples.append(sample)

        subsampled_clip_groups = {}
        total_clip_samples_before = sum(len(samples) for samples in clip_groups.values())
        total_clip_samples_after = 0

        for key, samples in clip_groups.items():
            sorted_samples = sorted(samples, key=lambda s: s['image_path'])
            subsampled = sorted_samples[::clip_subsample_rate]
            subsampled_clip_groups[key] = subsampled
            total_clip_samples_after += len(subsampled)

        sample_groups = []

        for key, samples in subsampled_clip_groups.items():
            sample_groups.append(samples)

        for sample in non_clip_samples:
            sample_groups.append([sample])

        random.shuffle(sample_groups)

        n_groups = len(sample_groups)
        n_train = max(1, int(n_groups * train_ratio))
        n_val = max(1, int(n_groups * val_ratio))

        train_groups = sample_groups[:n_train]
        val_groups = sample_groups[n_train:n_train + n_val]
        test_groups = sample_groups[n_train + n_val:]

        train_samples = [s for group in train_groups for s in group]
        val_samples = [s for group in val_groups for s in group]
        test_samples = [s for group in test_groups for s in group]

        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        def count_datasets(samples):
            counts = {}
            for s in samples:
                ds = s.get('dataset', 'unknown')
                counts[ds] = counts.get(ds, 0) + 1
            return counts

        train_counts = count_datasets(train_samples)
        val_counts = count_datasets(val_samples)
        test_counts = count_datasets(test_samples)

        print(f"\nDataset Random Split (clips kept together):")
        print(f"  Train: {len(train_samples)} samples - {train_counts}")
        print(f"  Val: {len(val_samples)} samples - {val_counts}")
        print(f"  Test: {len(test_samples)} samples - {test_counts}")

        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }


class MRZDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        img_size: int = 384,
        grayscale: bool = True,
        augment: bool = True,
        train: bool = True,
        aug_config: dict = None,
        use_kornia: bool = False
    ):
        self.samples = samples
        self.img_size = img_size
        self.grayscale = grayscale
        self.augment = augment
        self.train = train
        self.aug_config = aug_config or {}
        self.use_kornia = use_kornia

        if not use_kornia:
            if augment and train:
                self.transform = self._create_train_transform()
            else:
                self.transform = self._create_val_transform()
        else:
            self.transform = self._create_val_transform()

    def _create_train_transform(self):
        transforms = [
            A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR),
        ]

        if self.aug_config.get('aug_perspective', False):
            scale = self.aug_config.get('aug_perspective_scale', (0.05, 0.12))
            transforms.append(A.Perspective(scale=scale, p=0.95))

        if self.aug_config.get('aug_rotate', False):
            limit = self.aug_config.get('aug_rotate_limit', 25)
            transforms.append(A.Rotate(limit=limit, p=0.7, border_mode=cv2.BORDER_CONSTANT))

        if self.aug_config.get('aug_blur', False):
            blur_limit = self.aug_config.get('aug_blur_limit', 7)
            transforms.append(A.OneOf([
                A.MotionBlur(blur_limit=blur_limit, p=1.0),
                A.GaussianBlur(blur_limit=(3, blur_limit), p=1.0),
            ], p=0.5))

        if self.aug_config.get('aug_brightness_contrast', True):
            brightness = self.aug_config.get('aug_brightness_limit', 0.3)
            contrast = self.aug_config.get('aug_contrast_limit', 0.3)
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=brightness,
                contrast_limit=contrast,
                p=0.7
            ))

        if self.aug_config.get('aug_compression', False):
            quality = self.aug_config.get('aug_compression_quality', (60, 100))
            transforms.append(A.ImageCompression(quality_lower=quality[0], quality_upper=quality[1], p=0.5))

        return A.Compose(transforms, additional_targets={'mask': 'mask'})

    def _create_val_transform(self):
        return A.Compose([
            A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR),
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = cv2.imread(sample['image_path'])
        if img is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {sample['mask_path']}")

        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        if self.grayscale and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = img.astype(np.float32) / 255.0

        mask = (mask > 127).astype(np.float32)

        if self.grayscale:
            img_tensor = torch.from_numpy(img[None, ...])  # (1, H, W)
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)

        mask_tensor = torch.from_numpy(mask[None, ...])  # (1, H, W)

        return img_tensor, mask_tensor


def create_dataloaders(
    datasets: List[str],
    data_roots: Optional[Dict[str, str]] = None,
    batch_size: int = 16,
    img_size: int = 384,
    grayscale: bool = True,
    split_method: str = "random",
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    aug_config: dict = None,
    midv2020_modalities: Optional[List[str]] = None,
    midv500_conditions: Optional[List[str]] = None,
    clip_subsample_rate: int = 5,
    use_kornia: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if data_roots is None:
        data_roots = {
            "MIDV2020": "data/midv2020",
            "MIDV500": "data/midv500"
        }

    collectors = []

    for dataset_name in datasets:
        if dataset_name.upper() == "MIDV2020":
            root = data_roots.get("MIDV2020", "data/midv2020")
            print(f"\n[MIDV2020] Initializing collector from: {root}")
            collector = MIDV2020DataCollector(root, modalities=midv2020_modalities)
            collectors.append(collector)

        elif dataset_name.upper() == "MIDV500":
            root = data_roots.get("MIDV500", "data/midv500")
            print(f"\n[MIDV500] Initializing collector from: {root}")
            collector = MIDV500DataCollector(root, conditions=midv500_conditions)
            collectors.append(collector)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: MIDV2020, MIDV500")

    if not collectors:
        raise ValueError("No collectors created. Please specify at least one dataset.")

    multi_collector = MultiDatasetCollector(collectors)
    multi_collector.collect_all_samples(filter_passport=True)

    train_ratio, val_ratio, test_ratio = split_ratios

    if split_method == "by_document_type":
        splits = multi_collector.split_by_document_type(train_ratio, val_ratio, test_ratio, seed=seed)
    elif split_method == "random":
        splits = multi_collector.split_random_by_clips(train_ratio, val_ratio, test_ratio, seed=seed, clip_subsample_rate=clip_subsample_rate)
    else:
        raise ValueError(f"Unknown split_method: {split_method}. Use 'random' or 'by_document_type'")

    train_ds = MRZDataset(
        splits["train"],
        img_size=img_size,
        grayscale=grayscale,
        augment=True,
        train=True,
        aug_config=aug_config,
        use_kornia=use_kornia
    )

    val_ds = MRZDataset(
        splits["val"],
        img_size=img_size,
        grayscale=grayscale,
        augment=False,
        train=False,
        use_kornia=use_kornia
    )

    test_ds = MRZDataset(
        splits["test"],
        img_size=img_size,
        grayscale=grayscale,
        augment=False,
        train=False,
        use_kornia=use_kornia
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"\nDataLoaders created successfully!")
    print(f"  Datasets: {datasets}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Grayscale: {grayscale}")
    print(f"  Split method: {split_method}")
    print(f"  Workers: {num_workers}")

    return train_loader, val_loader, test_loader