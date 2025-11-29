#!/usr/bin/env python3
"""
Generate HED-MRZ training masks from MIDV-2020 annotations.

This script converts MIDV-2020 document quadrilateral annotations into
binary masks for MRZ (Machine Readable Zone) detection training.

The MRZ region is computed using homography transformation from template
annotations to perspective-distorted frames, ensuring accurate MRZ localization.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm


class MRZMaskGenerator:
    """Generate MRZ masks using template-based homography transformation."""

    # Only these document types have MRZ annotations in templates
    MRZ_DOCUMENT_TYPES = [
        'aze_passport',
        'grc_passport',
        'lva_passport',
        'srb_passport'
    ]

    # Dilation for mask (pixels)
    DILATION_PX = 3

    def __init__(self, base_path: str):
        """
        Initialize the mask generator.

        Args:
            base_path: Path to MIDV-2020 dataset root
        """
        self.base_path = Path(base_path)
        self.templates_dir = self.base_path / 'templates'
        self.template_mrz_cache = {}  # Cache template MRZ regions by doc_type
        self.stats = {
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'no_mrz': 0
        }

    def load_template_mrz_regions(self, doc_type: str) -> Dict[str, np.ndarray]:
        """
        Load MRZ regions from template annotations.

        Args:
            doc_type: Document type name (e.g., 'aze_passport')

        Returns:
            Dictionary mapping filename to MRZ bounding box polygon (4 corners)
            Returns empty dict if document type has no MRZ
        """
        if doc_type in self.template_mrz_cache:
            return self.template_mrz_cache[doc_type]

        # Check if this document type has MRZ
        if doc_type not in self.MRZ_DOCUMENT_TYPES:
            self.template_mrz_cache[doc_type] = {}
            return {}

        # Load template annotations
        template_ann_path = self.templates_dir / 'annotations' / f'{doc_type}.json'

        if not template_ann_path.exists():
            print(f"  Warning: Template annotations not found: {template_ann_path}")
            self.template_mrz_cache[doc_type] = {}
            return {}

        with open(template_ann_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        via_metadata = template_data.get('_via_img_metadata', {})
        mrz_regions = {}

        # Extract MRZ regions for each template image
        for img_key, img_data in via_metadata.items():
            filename = img_data.get('filename')
            if not filename:
                continue

            # Find MRZ line regions
            mrz_line0 = None
            mrz_line1 = None

            for region in img_data.get('regions', []):
                attrs = region.get('region_attributes', {})
                field_name = attrs.get('field_name', '')

                if field_name == 'mrz_line0':
                    mrz_line0 = region
                elif field_name == 'mrz_line1':
                    mrz_line1 = region

            # Skip if no MRZ found
            if mrz_line0 is None and mrz_line1 is None:
                continue

            # Combine both MRZ lines into single bounding box
            mrz_rects = []
            for mrz_line in [mrz_line0, mrz_line1]:
                if mrz_line is not None:
                    shape = mrz_line['shape_attributes']
                    if shape.get('name') == 'rect':
                        x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
                        mrz_rects.append((x, y, x + w, y + h))

            if not mrz_rects:
                continue

            # Compute bounding box of all MRZ lines
            min_x = min(r[0] for r in mrz_rects)
            min_y = min(r[1] for r in mrz_rects)
            max_x = max(r[2] for r in mrz_rects)
            max_y = max(r[3] for r in mrz_rects)

            # Convert to 4-corner polygon (clockwise from top-left)
            mrz_polygon = np.array([
                [min_x, min_y],  # top-left
                [max_x, min_y],  # top-right
                [max_x, max_y],  # bottom-right
                [min_x, max_y]   # bottom-left
            ], dtype=np.float32)

            mrz_regions[filename] = mrz_polygon

        self.template_mrz_cache[doc_type] = mrz_regions
        return mrz_regions

    def get_template_image_dimensions(self, doc_type: str, filename: str) -> Optional[Tuple[int, int]]:
        """
        Get dimensions of a template image.

        Args:
            doc_type: Document type name
            filename: Image filename

        Returns:
            (height, width) tuple or None if image not found
        """
        template_img_path = self.templates_dir / 'images' / doc_type / filename

        if not template_img_path.exists():
            return None

        img = cv2.imread(str(template_img_path))
        if img is None:
            return None

        return img.shape[:2]  # (height, width)

    def compute_homography(
        self,
        template_dims: Tuple[int, int],
        frame_quad: np.ndarray
    ) -> np.ndarray:
        """
        Compute homography matrix from template to frame.

        Args:
            template_dims: Template image dimensions (height, width)
            frame_quad: Frame document quadrilateral (4 corners, clockwise from top-left)

        Returns:
            3x3 homography matrix
        """
        height, width = template_dims

        # Template document corners (assume full image is the document)
        # Clockwise from top-left: TL, TR, BR, BL
        template_quad = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)

        # Compute homography
        H, _ = cv2.findHomography(template_quad, frame_quad)

        return H

    def transform_mrz_region(
        self,
        mrz_polygon: np.ndarray,
        homography: np.ndarray
    ) -> np.ndarray:
        """
        Transform MRZ polygon from template space to frame space.

        Args:
            mrz_polygon: MRZ polygon in template coordinates (4 corners)
            homography: 3x3 homography matrix

        Returns:
            Transformed MRZ polygon in frame coordinates
        """
        # Reshape for cv2.perspectiveTransform (needs shape [N, 1, 2])
        mrz_points = mrz_polygon.reshape(-1, 1, 2)

        # Apply homography
        transformed_points = cv2.perspectiveTransform(mrz_points, homography)

        # Reshape back to [N, 2]
        return transformed_points.reshape(-1, 2)

    def generate_mask(
        self,
        mrz_polygon: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Generate binary mask from MRZ polygon.

        Args:
            mrz_polygon: MRZ polygon as [N, 2] array
            image_shape: Image dimensions as (height, width)

        Returns:
            Binary mask (uint8) with 255 inside MRZ, 0 outside
        """
        # Create blank mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to integer coordinates
        pts = mrz_polygon.astype(np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [pts], 255)

        # Apply dilation to make mask slightly larger
        if self.DILATION_PX > 0:
            kernel = np.ones((self.DILATION_PX, self.DILATION_PX), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def process_annotation_file(
        self,
        annotation_path: Path,
        images_dir: Path,
        output_dir: Path,
        doc_type: str,
        clip_template_name: str = None
    ) -> None:
        """
        Process a single annotation JSON file.

        Args:
            annotation_path: Path to VIA JSON annotation file
            images_dir: Directory containing source images
            output_dir: Directory to save generated masks
            doc_type: Document type name
            clip_template_name: For clips modality, the template filename to use (e.g., "00.jpg")
                               If provided, this template is used for all frames instead of matching by filename
        """
        # Load template MRZ regions for this document type
        template_mrz_regions = self.load_template_mrz_regions(doc_type)

        if not template_mrz_regions:
            print(f"  Skipping {doc_type} (no MRZ annotations)")
            self.stats['no_mrz'] += 1
            return

        # Load frame annotations
        with open(annotation_path, 'r', encoding='utf-8') as f:
            frame_data = json.load(f)

        via_metadata = frame_data.get('_via_img_metadata', {})

        if not via_metadata:
            print(f"  Warning: No image metadata in {annotation_path}")
            return

        # Process each image
        for img_key, img_data in tqdm(via_metadata.items(),
                                       desc=f"  {doc_type}",
                                       leave=False):
            filename = img_data.get('filename')

            if not filename:
                self.stats['skipped'] += 1
                continue

            # Determine which template to use
            if clip_template_name:
                # For clips, use the specified template for all frames
                template_filename = clip_template_name
            else:
                # For other modalities, use the frame's own filename
                template_filename = filename

            # Check if template has MRZ for this image
            if template_filename not in template_mrz_regions:
                # print(f"    Warning: No template MRZ for {template_filename}, skipping")
                self.stats['skipped'] += 1
                continue

            # Get template MRZ polygon
            template_mrz_polygon = template_mrz_regions[template_filename]

            # Get template image dimensions
            template_dims = self.get_template_image_dimensions(doc_type, template_filename)
            if template_dims is None:
                print(f"    Warning: Could not load template image for {filename}")
                self.stats['skipped'] += 1
                continue

            # Find document quadrilateral region in frame
            doc_quad_region = None
            for region in img_data.get('regions', []):
                attrs = region.get('region_attributes', {})
                if attrs.get('field_name') == 'doc_quad':
                    doc_quad_region = region
                    break

            if not doc_quad_region:
                print(f"    Warning: No doc_quad for {filename}, skipping")
                self.stats['skipped'] += 1
                continue

            # Extract quadrilateral coordinates
            shape_attrs = doc_quad_region['shape_attributes']
            if shape_attrs.get('name') != 'polygon':
                print(f"    Warning: doc_quad is not a polygon for {filename}")
                self.stats['skipped'] += 1
                continue

            all_x = shape_attrs['all_points_x']
            all_y = shape_attrs['all_points_y']

            if len(all_x) != 4 or len(all_y) != 4:
                print(f"    Warning: doc_quad has {len(all_x)} points (expected 4) for {filename}")
                self.stats['skipped'] += 1
                continue

            # Create frame doc_quad array (clockwise from top-left)
            frame_quad = np.array([[x, y] for x, y in zip(all_x, all_y)], dtype=np.float32)

            # Load frame image to get dimensions
            image_path = images_dir / filename

            if not image_path.exists():
                print(f"    Warning: Image not found: {image_path}")
                self.stats['skipped'] += 1
                continue

            try:
                # Load image to get dimensions
                img_cv = cv2.imread(str(image_path))
                if img_cv is None:
                    print(f"    Warning: Could not load image: {image_path}")
                    self.stats['skipped'] += 1
                    continue

                image_shape = img_cv.shape[:2]  # (height, width)

                # Compute homography from template to frame
                homography = self.compute_homography(template_dims, frame_quad)

                if homography is None:
                    print(f"    Warning: Could not compute homography for {filename}")
                    self.stats['skipped'] += 1
                    continue

                # Transform MRZ polygon from template space to frame space
                frame_mrz_polygon = self.transform_mrz_region(template_mrz_polygon, homography)

                # Generate mask
                mask = self.generate_mask(frame_mrz_polygon, image_shape)

                # Save mask (change extension to .png)
                mask_filename = Path(filename).stem + '.png'
                mask_path = output_dir / mask_filename

                cv2.imwrite(str(mask_path), mask)

                self.stats['processed'] += 1

            except Exception as e:
                print(f"    Error processing {filename}: {e}")
                self.stats['errors'] += 1

    def process_modality_directory(self, modality: str) -> None:
        """
        Process all annotations in a modality directory (e.g., 'photo').

        Args:
            modality: Modality name ('photo', 'scan_upright', 'scan_rotated')
        """
        modality_path = self.base_path / modality
        annotations_dir = modality_path / 'annotations'
        images_base_dir = modality_path / 'images'

        if not annotations_dir.exists():
            print(f"Warning: Annotations directory not found: {annotations_dir}")
            return

        print(f"\nProcessing {modality}/")

        # Find all JSON annotation files
        annotation_files = list(annotations_dir.glob('*.json'))

        if not annotation_files:
            print(f"  No annotation files found in {annotations_dir}")
            return

        # Process each annotation file
        for ann_file in annotation_files:
            doc_type = ann_file.stem  # e.g., 'aze_passport'

            # Only process documents with MRZ
            if doc_type not in self.MRZ_DOCUMENT_TYPES:
                continue

            images_dir = images_base_dir / doc_type

            if not images_dir.exists():
                print(f"  Warning: Images directory not found: {images_dir}")
                continue

            # Create output directory
            output_base = modality_path / 'annotations-hed-mrz'
            output_dir = output_base / doc_type
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process this annotation file
            self.process_annotation_file(ann_file, images_dir, output_dir, doc_type)

    def process_clips_modality(self) -> None:
        """
        Process clips modality with its nested structure.

        Clips has a different directory structure:
        - annotations/{doc_type}/{clip_number}.json
        - images/{doc_type}/{clip_number}/
        - output: annotations-hed-mrz/{doc_type}/{clip_number}/
        """
        modality_path = self.base_path / 'clips'
        annotations_base = modality_path / 'annotations'
        images_base = modality_path / 'images'

        if not annotations_base.exists():
            print(f"Warning: Clips annotations directory not found: {annotations_base}")
            return

        print(f"\nProcessing clips/")

        # Iterate through document types
        doc_types = [d for d in annotations_base.iterdir() if d.is_dir()]

        if not doc_types:
            print(f"  No document types found in {annotations_base}")
            return

        for doc_type_dir in doc_types:
            doc_type = doc_type_dir.name

            # Only process documents with MRZ
            if doc_type not in self.MRZ_DOCUMENT_TYPES:
                continue

            # Find all clip annotation files
            clip_annotations = list(doc_type_dir.glob('*.json'))

            if not clip_annotations:
                print(f"  No clip annotations found for {doc_type}")
                continue

            # Process each clip
            for clip_ann_file in tqdm(clip_annotations,
                                       desc=f"  {doc_type}",
                                       leave=False):
                clip_number = clip_ann_file.stem  # e.g., '00', '01'

                # Images are in images/{doc_type}/{clip_number}/
                images_dir = images_base / doc_type / clip_number

                if not images_dir.exists():
                    print(f"    Warning: Images directory not found: {images_dir}")
                    continue

                # Output to annotations-hed-mrz/{doc_type}/{clip_number}/
                output_dir = modality_path / 'annotations-hed-mrz' / doc_type / clip_number
                output_dir.mkdir(parents=True, exist_ok=True)

                # For clips, use the template with the same number as the clip
                # e.g., clip "00" uses template "00.jpg"
                clip_template_name = f"{clip_number}.jpg"

                # Process this clip's annotation file
                self.process_annotation_file(clip_ann_file, images_dir, output_dir, doc_type,
                                             clip_template_name=clip_template_name)

    def run(self, modalities: Optional[List[str]] = None) -> None:
        """
        Run the mask generation for specified modalities.

        Args:
            modalities: List of modalities to process. If None, processes
                       ['photo', 'scan_upright', 'scan_rotated', 'clips']
        """
        if modalities is None:
            modalities = ['photo', 'scan_upright', 'scan_rotated', 'clips']

        print(f"Starting HED-MRZ mask generation (homography-based)")
        print(f"Dataset path: {self.base_path}")
        print(f"Modalities: {', '.join(modalities)}")
        print(f"Processing only MRZ documents: {', '.join(self.MRZ_DOCUMENT_TYPES)}")
        print(f"Dilation: {self.DILATION_PX}px")

        # Process each modality
        for modality in modalities:
            if modality == 'clips':
                # Clips has a different directory structure
                self.process_clips_modality()
            else:
                self.process_modality_directory(modality)

        # Print statistics
        print("\n" + "="*60)
        print("Processing complete!")
        print(f"  Masks generated: {self.stats['processed']}")
        print(f"  Skipped (no template MRZ or missing data): {self.stats['skipped']}")
        print(f"  Document types without MRZ (skipped): {self.stats['no_mrz']}")
        print(f"  Errors: {self.stats['errors']}")
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate HED-MRZ training masks from MIDV-2020 annotations using homography'
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        choices=['photo', 'scan_upright', 'scan_rotated', 'clips'],
        help='Modalities to process (default: all)'
    )

    args = parser.parse_args()

    # Determine dataset path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_path = project_root / 'data' / 'midv2020'

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure MIDV-2020 dataset is downloaded and extracted.")
        sys.exit(1)

    # Check templates directory
    templates_path = dataset_path / 'templates'
    if not templates_path.exists():
        print(f"Error: Templates directory not found at {templates_path}")
        print("Templates are required for homography-based MRZ mask generation.")
        sys.exit(1)

    # Create generator and run
    generator = MRZMaskGenerator(dataset_path)
    generator.run(modalities=args.modalities)


if __name__ == '__main__':
    main()
