#!/bin/bash
# Script to download and extract only image archives from MIDV-2020 via FTP

set -e

BASE_URL="ftp://smartengines.com/midv-2020/dataset"
TARGET="."
TAR_DIR="$TARGET/tar"
IMAGE_DIR="$TARGET/"

# Relevante .tar-Archive mit Bildern (keine Videos)
# TARS=(
#  "templates.tar"
#  "scan_upright.tar"
#  "clips.tar"
#  "photo.tar"
#  "scan_rotated.tar"
# )

TARS=(
 "photo.tar"
)

echo "📁 Setting up directories..."
mkdir -p "$TAR_DIR" "$IMAGE_DIR"

echo "🔽 Downloading .tar files from $BASE_URL..."
for tar_file in "${TARS[@]}"; do
  if [ -f "$TAR_DIR/$tar_file" ]; then
    echo "✅ $tar_file already exists — skipping download"
  else
    echo "➡️ Downloading $tar_file"
    lftp "$BASE_URL" -e "get $tar_file -o $TAR_DIR/$tar_file; bye"
  fi
done

echo "🗜️ Extracting each archive into its own subdirectory..."
for tar_file in "${TARS[@]}"; do
  folder_name="${tar_file%.tar}"  # Remove .tar
  extract_path="$IMAGE_DIR/$folder_name"
  mkdir -p "$extract_path"
  echo "➡️ Extracting $tar_file into $extract_path"
  tar -xf "$TAR_DIR/$tar_file" -C "$extract_path"
  echo "🧹 Deleting $tar_file"
  rm "$TAR_DIR/$tar_file"
done

echo "📊 Counting all .jpg/.jpeg files across all subfolders..."
NUM=$(find "$IMAGE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)
SIZE=$(du -sh "$IMAGE_DIR" | cut -f1)

rm -r "$TAR_DIR"

echo "✅ Done!"
echo "🖼️ Total images: $NUM files"
echo "📦 Total size: $SIZE"
echo "📂 All extracted in: $IMAGE_DIR/"
