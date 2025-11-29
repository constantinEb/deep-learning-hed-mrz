#!/usr/bin/env python3
"""
Generate HED-MRZ training masks from MIDV-500 using proper homography transformation.

Uses template images and annotations to transform MRZ regions onto instance images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


class MRZMaskGeneratorMIDV500:
    """Generate MRZ masks using homography from template to instances."""

    CAPTURE_CONDITIONS = ['CA', 'CS', 'HA', 'HS', 'KA', 'KS', 'PA', 'PS', 'TA', 'TS']
    DILATION_PX = 3

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.template_cache = {}
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'no_mrz': 0,
            'doc_types_with_mrz': []
        }

    def load_template_data(self, doc_type: str) -> Optional[dict]:
        """Load template image dimensions and unified MRZ quad."""
        if doc_type in self.template_cache:
            return self.template_cache[doc_type]

        # Load template image
        template_img_path = self.base_path / doc_type / 'images' / f'{doc_type}.tif'
        if not template_img_path.exists():
            # Try other extensions
            for ext in ['.jpg', '.png']:
                template_img_path = self.base_path / doc_type / 'images' / f'{doc_type}{ext}'
                if template_img_path.exists():
                    break
            else:
                self.template_cache[doc_type] = None
                return None

        template_img = cv2.imread(str(template_img_path))
        if template_img is None:
            self.template_cache[doc_type] = None
            return None

        template_height, template_width = template_img.shape[:2]

        # Load template annotations
        template_ann_path = self.base_path / doc_type / 'ground_truth' / f'{doc_type}.json'
        if not template_ann_path.exists():
            self.template_cache[doc_type] = None
            return None

        with open(template_ann_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        # Find MRZ fields (contain '<<' characters)
        mrz_field_quads = []
        for field_name, field_data in template_data.items():
            if not isinstance(field_data, dict):
                continue
            value = field_data.get('value', '')
            quad = field_data.get('quad')
            if isinstance(value, str) and '<<' in value and quad:
                mrz_field_quads.append(np.array(quad, dtype=np.float32))

        if not mrz_field_quads:
            self.template_cache[doc_type] = None
            return None

        # Create unified MRZ quad spanning all lines
        # Assume quads are ordered: TL, TR, BR, BL
        # Find topmost and bottommost lines based on Y coordinates
        top_line = min(mrz_field_quads, key=lambda q: min(q[:, 1]))  # Line with smallest Y
        bottom_line = max(mrz_field_quads, key=lambda q: max(q[:, 1]))  # Line with largest Y

        # Create unified quad:
        # Top-left and top-right from top line
        # Bottom-right and bottom-left from bottom line
        unified_mrz_quad = np.array([
            top_line[0],     # Top-left from top line
            top_line[1],     # Top-right from top line
            bottom_line[2],  # Bottom-right from bottom line
            bottom_line[3]   # Bottom-left from bottom line
        ], dtype=np.float32)

        result = {
            'template_width': template_width,
            'template_height': template_height,
            'mrz_quad': unified_mrz_quad
        }

        self.template_cache[doc_type] = result
        return result

    def compute_homography(self, template_width: int, template_height: int, instance_quad: np.ndarray) -> np.ndarray:
        """
        Compute homography from template image to instance document quad.
        
        Args:
            template_width: Width of template image
            template_height: Height of template image
            instance_quad: Instance document quad (4 corners)
        
        Returns:
            3x3 homography matrix
        """
        # Template image corners (full image = document)
        template_quad = np.array([
            [0, 0],                            # top-left
            [template_width, 0],               # top-right
            [template_width, template_height], # bottom-right
            [0, template_height]               # bottom-left
        ], dtype=np.float32)

        # Compute homography
        H, _ = cv2.findHomography(template_quad, instance_quad)
        return H

    def transform_quads(self, quads: List[np.ndarray], homography: np.ndarray) -> List[np.ndarray]:
        """Transform list of quads using homography."""
        transformed = []
        for quad in quads:
            points = quad.reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(points, homography)
            transformed.append(transformed_points.reshape(-1, 2))
        return transformed

    def generate_mask(self, quads: List[np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate binary mask from list of quads."""
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Fill each quad
        for quad in quads:
            pts = quad.astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

        # Apply dilation
        if self.DILATION_PX > 0:
            kernel = np.ones((self.DILATION_PX, self.DILATION_PX), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def process_document_type(self, doc_type: str) -> None:
        """Process a single document type."""
        doc_type_path = self.base_path / doc_type

        if not doc_type_path.exists():
            return

        # Load template data
        template_data = self.load_template_data(doc_type)
        if template_data is None:
            self.stats['no_mrz'] += 1
            return

        template_width = template_data['template_width']
        template_height = template_data['template_height']
        mrz_quad = template_data['mrz_quad']

        if doc_type not in self.stats['doc_types_with_mrz']:
            self.stats['doc_types_with_mrz'].append(doc_type)

        # Create output directory
        output_dir = doc_type_path / 'annotations-hed-mrz'
        output_dir.mkdir(parents=True, exist_ok=True)

        images_dir = doc_type_path / 'images'
        ground_truth_dir = doc_type_path / 'ground_truth'

        for condition in self.CAPTURE_CONDITIONS:
            condition_images_dir = images_dir / condition
            condition_gt_dir = ground_truth_dir / condition

            if not condition_images_dir.exists() or not condition_gt_dir.exists():
                continue

            instance_files = sorted(condition_gt_dir.glob(f'{condition}*.json'))

            for instance_file in instance_files:
                instance_name = instance_file.stem

                try:
                    # Load instance annotation
                    with open(instance_file, 'r', encoding='utf-8') as f:
                        instance_data = json.load(f)

                    instance_quad = instance_data.get('quad')
                    if not instance_quad or len(instance_quad) != 4:
                        self.stats['skipped'] += 1
                        continue

                    instance_quad = np.array(instance_quad, dtype=np.float32)

                    # Find corresponding image
                    image_path = None
                    for ext in ['.tif', '.jpg', '.png']:
                        candidate = condition_images_dir / f'{instance_name}{ext}'
                        if candidate.exists():
                            image_path = candidate
                            break

                    if not image_path:
                        self.stats['skipped'] += 1
                        continue

                    # Load image
                    img = cv2.imread(str(image_path))
                    if img is None:
                        self.stats['skipped'] += 1
                        continue

                    image_shape = img.shape[:2]

                    # Compute homography from template to instance
                    homography = self.compute_homography(template_width, template_height, instance_quad)
                    if homography is None:
                        self.stats['skipped'] += 1
                        continue

                    # Transform unified MRZ quad
                    transformed_mrz_quad = self.transform_quads([mrz_quad], homography)[0]

                    # Generate mask
                    mask = self.generate_mask([transformed_mrz_quad], image_shape)

                    # Save mask
                    mask_path = output_dir / f'{instance_name}.png'
                    cv2.imwrite(str(mask_path), mask)

                    self.stats['processed'] += 1

                except Exception as e:
                    print(f"    Error processing {instance_name}: {e}")
                    self.stats['errors'] += 1

    def run(self, doc_types: Optional[List[str]] = None) -> None:
        """Run mask generation."""
        print(f"Starting HED-MRZ mask generation for MIDV-500")
        print(f"Dataset path: {self.base_path}")
        print(f"Using homography transformation from template images")
        print(f"Creating unified masks spanning all MRZ lines")
        print(f"Auto-detecting MRZ fields by '<<' characters")
        print(f"Processing all capture conditions: {', '.join(self.CAPTURE_CONDITIONS)}")
        print(f"Dilation: {self.DILATION_PX}px")

        if doc_types is None:
            doc_types = [d.name for d in self.base_path.iterdir() if d.is_dir()]
            doc_types.sort()

        print(f"\nScanning {len(doc_types)} document types...")

        for doc_type in tqdm(doc_types, desc="Processing"):
            self.process_document_type(doc_type)

        print("\n" + "="*60)
        print("Processing complete!")
        print(f"  Document types with MRZ: {len(self.stats['doc_types_with_mrz'])}")
        if self.stats['doc_types_with_mrz']:
            print(f"    {', '.join(self.stats['doc_types_with_mrz'])}")
        print(f"  Masks generated: {self.stats['processed']}")
        print(f"  Skipped (missing data): {self.stats['skipped']}")
        print(f"  Document types without MRZ: {self.stats['no_mrz']}")
        print(f"  Errors: {self.stats['errors']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate HED-MRZ masks from MIDV-500 using homography'
    )
    parser.add_argument('--doc-types', nargs='+', help='Document types to process')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_path = project_root / 'data' / 'midv500'

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    generator = MRZMaskGeneratorMIDV500(dataset_path)
    generator.run(doc_types=args.doc_types)


if __name__ == '__main__':
    main()
